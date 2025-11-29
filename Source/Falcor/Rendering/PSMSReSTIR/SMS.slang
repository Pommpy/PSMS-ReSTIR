#include "Utils/Math/MathConstants.slangh"

import Scene.RaytracingInline;
import Utils.Sampling.SampleGenerator;
import Rendering.PSMSReSTIR.LoadShadingData;
import Rendering.Materials.Fresnel;
import Rendering.PSMSReSTIR.StaticParams;
import Reservoir;
import Utils.Debug.PixelDebug;

static const uint maxBounces = 3;
static const float uniqueness_threshold = 1e-4f;
static const uint maxIterations = 20;
static const float rayEpsilon = FLT_EPSILON * 750;

#ifdef SOLVER_THRESHOLD
static const float solverThreshold = kSolverThreshold;
#else
static const float solverThreshold = 1e-4f;
#endif

struct ManifoldVertex
{
    // Position and partials
    float3 p = float3(0.f);
    float3 dp_du = float3(0.f);
    float3 dp_dv = float3(0.f);

    // Normal and partials
    float3 n = float3(0.f), gn = float3(0.f);
    float3 dn_du = float3(0.f), dn_dv = float3(0.f);

    // Tangents and partials
    //float3 s, t;
    //float3 ds_du, ds_dv;
    //float3 dt_du, dt_dv;

    // Further information
    float eta = 1.f;
    float2 uv = float2(0.f);
    uint materialID = 0;
    //ShapePtr shape;
    bool fixedDirection = false;
    // Used in multi-bounce version
    float2 C = float2(0.f);
    float2x2 dC_dx_prev = float2x2(0.f), dC_dx_cur = float2x2(0.f), dC_dx_next = float2x2(0.f);
    float2x2 tmp = float2x2(0.f), inv_lambda = float2x2(0.f);
    float2 dx = float2(0.f);

    __init(const ShadingData sd)
    {
        this.p = sd.posW;
        this.gn = sd.faceN;
        this.n = sd.frame.N;
        float16_t IoR = sd.mtl.getIoR();

        // incident / transmitted
        this.eta = sd.frontFacing ? (sd.IoR / IoR) : (IoR / sd.IoR);
        this.n = sd.frontFacing ? sd.frame.N : -sd.frame.N;
        this.uv = sd.uv;
        this.materialID = sd.materialID;
        this.fixedDirection = false;
    }

    [mutating]
    void setup(const ShadingData sd)
    {
        this.p = sd.posW;
        this.gn = sd.faceN;
        this.n = sd.frame.N;
        float16_t IoR = sd.mtl.getIoR();
        this.eta = sd.frontFacing ? (sd.IoR / IoR) : (IoR / sd.IoR);
        this.n = sd.frontFacing ? sd.frame.N : -sd.frame.N;

        this.uv = sd.uv;
        this.materialID = sd.materialID;
        this.fixedDirection = false;
    }

    [mutating]
    void makeOrthonormal() {
        // Turn into orthonormal parameterization at 'p'
        float l = length(dp_du);
        if(l == 0) return;
        float inv_norm = rcp(l);
        dp_du *= inv_norm;
        dn_du *= inv_norm;
        float dp = dot(dp_du, dp_dv);
        float3 dp_dv_tmp = dp_dv - dp * dp_du;
        float3 dn_dv_tmp = dn_dv - dp * dn_du;
        l = length(dp_dv_tmp);
        if(l == 0) return;
        inv_norm = rcp(l);
        dp_dv = dp_dv_tmp * inv_norm;
        dn_dv = dn_dv_tmp * inv_norm;
    }
}

struct SMS
{
    StructuredBuffer<AABB> specularAABBs;

    StructuredBuffer<uint> specularMaterialIDs;
    StructuredBuffer<bool> isUVSpaceSampling;
    uint specularShapesCount;
    int numTilesX;

    float3 samplePos(float2 uv, uint specularIndex, inout TextureHandle positionMap, inout TextureHandle shadingNormalMap)
    {
        uint materialID = specularMaterialIDs[specularIndex];
        let material = gScene.materials.getMaterial(materialID);
        material.getPositionAndNormalMap(positionMap, shadingNormalMap);
        SamplerState s = gScene.materials.getDefaultTextureSampler(materialID);
        return gScene.materials.sampleTexture(positionMap, s, uv, 0.f).xyz;
    }
    float2 dirToCanonical(float3 dir)
    {
        float cosTheta = min(max(dir.y, -1.f), 1.f);
        float phi = atan2(dir.z, dir.x);
        if (phi < 0.f) phi += 2.f * M_PI;
        return float2((cosTheta + 1.f) / 2, phi / (2 * M_PI));
    }
    float3 canonicalToDir(float2 uv)
    {
        float cosTheta = 2.f * uv.x - 1.f;
        float sinTheta = sqrt(1.f - cosTheta * cosTheta);
        float phi = 2.f * M_PI * uv.y;
        return float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
    }
    float2 uvFromBoundToDirSpace(float2 uvInBound, float2 uvMin, float2 uvMax)
    {
        float2 size = uvMax - uvMin;
        float2 uv = uvMin + uvInBound * size;
        if (uv.y > 1.f) uv.y -= 1.f;
        return uv;
    }
    float2 uvFromDirSpaceToBound(float2 uvInDir, float2 uvMin, float2 uvMax)
    {
        float2 uvInBound = uvInDir - uvMin;
        if (uvInBound.y < 0.f) uvInBound.y += 1.f;
        return uvInBound / (uvMax - uvMin);
    }
    void loosenBound(inout float2 uvMin, inout float2 uvMax, float alpha)
    {
        uvMin -= float2(alpha);
        uvMax += float2(alpha);
        uvMin = max(uvMin, float2(0.f));
        uvMax.x = min(uvMax.x, 1.f);
        uvMax.y = min(uvMax.y, uvMin.y + 1.f);
    }
    void getDirectionSpaceBound(float3 pos, AABB aabb, inout float2 uvMin, inout float2 uvMax)
    {
        // get cos theta bound
        float cosMin = 1.f, cosMax = -1.f;
        
        float2 closestPoint = clamp(pos.xz, aabb.minPoint.xz, aabb.maxPoint.xz);
        float2 xzMinDelta = closestPoint - pos.xz;
        float dMin2 = dot(xzMinDelta, xzMinDelta);
        // find max dist in xy plane
        float2 xzMaxDelta = float2(0.f);
        xzMaxDelta.x = max(abs(aabb.minPoint.x - pos.x), abs(aabb.maxPoint.x - pos.x));
        xzMaxDelta.y = max(abs(aabb.minPoint.z - pos.z), abs(aabb.maxPoint.z - pos.z));
        float dMax2 = dot(xzMaxDelta, xzMaxDelta);
        for (int i = 0; i < 2; i++)
        {
            float y = (i == 0) ? aabb.minPoint.y : aabb.maxPoint.y;
            y = y - pos.y;
            float y2 = y * y;
            float a = y / sqrt(dMin2 + y2);
            float b = y / sqrt(dMax2 + y2);
            cosMin = min(cosMin, a);
            cosMin = min(cosMin, b);
            cosMax = max(cosMax, a);
            cosMax = max(cosMax, b);
        }
        // find largest cover
        float phiStart = 0.f;
        float phiEnd = M_2PI;
        int cnt = 0;
        float phis[4];
        for (int i = 0; i < 4; i++)
        {
            float2 p = float2((i & 1)? aabb.minPoint.x : aabb.maxPoint.x, (i & 2)? aabb.minPoint.z : aabb.maxPoint.z) - float2(pos.x, pos.z);
            float phi = atan2(p.y, p.x);
            while (phi < 0.f) phi += M_2PI;
            phis[cnt++] = phi;
        }
        // bubble sort
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3 - i; j++)
            {
                if (phis[j] > phis[j + 1])
                {
                    float tmp = phis[j];
                    phis[j] = phis[j + 1];
                    phis[j + 1] = tmp;
                }
            }
        }
        // bool inside = all(pos.xy >= aabb.minPoint.xy && pos.xy <= aabb.maxPoint.xy);
        float maxGap = -1.f;
        int maxIdx = 0;
        for (int i = 0; i < 4; i++)
        {
            int j = i == 3? 0 : i + 1;
            float gap = (j == 0) ? phis[0] + M_2PI - phis[i] : phis[j] - phis[i];
            if(gap > maxGap)
            {
                maxGap = gap;
                maxIdx = i;
            }
        }
        phiStart = phis[(maxIdx + 1) % 4];
        phiEnd = phis[maxIdx];
        if(maxGap < M_PI)
        {
            phiStart = 0.f;
            phiEnd = 2.f * M_PI;
        }
        uvMin = float2((cosMin + 1.f) / 2, phiStart / M_2PI);
        uvMax = float2((cosMax + 1.f) / 2, phiEnd / M_2PI);
        if (uvMin.y > uvMax.y)
            uvMax.y += 1.f;
    }
    float3 sampleDir(float3 pos, float2 uv, uint specularIndex)
    {
        AABB aabb = specularAABBs[specularIndex];
        float2 uvMin, uvMax;
        getDirectionSpaceBound(pos, aabb, uvMin, uvMax);
        float2 d = uvMax - uvMin;
        uv = uvMin + d * uv;
        if(uv.y > 1.f) uv.y -= 1.f;
        return canonicalToDir(uv);
    }

    uint2 invProbEstimation<S : ISampleGenerator, let directional : bool>( float3 siPos, float3 lightPosOrDir, float3 direction, inout S sg, inout uint bounces, inout ManifoldVertex currentPath[], inout uint seedMaterialID[])
    {
        float3 tmp = float3(0.f);
        uint invProbEstimate = 0;
        uint newton_iterations = 0;
        while (invProbEstimate < kMaxBernoulliTrials)
        {
            invProbEstimate++;
            int2 success_counter = samplePath<S, directional>(siPos, lightPosOrDir, sg, false, currentPath, seedMaterialID, bounces, tmp);
            newton_iterations += success_counter.y;
            bool successTrial = success_counter.x == 1;
            float3 directionTrial = normalize(currentPath[0].p - siPos);
            if (successTrial && abs(dot(direction, directionTrial) - 1.f) < uniqueness_threshold)
            {
                break;
            }
        }
        return uint2(invProbEstimate, newton_iterations);
    }

    float3 specularManifoldSampling<S : ISampleGenerator, let directional : bool>(const ManifoldVertex si, float3 lightPosOrDir, inout S sg, inout float3 specularPos, out uint3 invProbEstimateNewtons)
    {
        // specularPos = {};
        invProbEstimateNewtons = uint3(0);

        uint newton_iterations = 0;
        if (specularShapesCount == 0)
        {
            return float3(0.f);
        }

        ManifoldVertex currentPath[maxBounces];
        uint seedMaterialID[maxBounces];
        uint bounces;
        int2 success_counter = samplePath<S, directional>(si.p, lightPosOrDir, sg, true, currentPath, seedMaterialID, bounces, specularPos);
        newton_iterations += success_counter.y;
        invProbEstimateNewtons.y = newton_iterations;
        bool success = success_counter.x == 1;
        if (!success)
        {
            return float3(0.f);
        }
        specularPos = currentPath[0].p;
        float3 direction = normalize(currentPath[0].p - si.p);

        float3 specularVal;
        switch (bounces)
        {
        case 1:
            specularVal = evalSpecularContribution<1, directional>(si, lightPosOrDir, currentPath);
            break;
        case 2:
            specularVal = evalSpecularContribution<2, directional>(si, lightPosOrDir, currentPath);
            break;
        default:
            specularVal = float3(0.f);
        }
        if (all(specularVal == 0.f)) return float3(0.f);
        uint2 inv_newtons = invProbEstimation<S, directional>(si.p, lightPosOrDir, direction, sg, bounces, currentPath, seedMaterialID);
        invProbEstimateNewtons.x = inv_newtons.x;
        invProbEstimateNewtons.z = newton_iterations + inv_newtons.y;
        return specularVal;
    }

    float3 specularManifoldResampling<S : ISampleGenerator, let directional : bool>(const ShadingData sd, const ShadingDifferentialGeometry sdg, const IMaterialInstance mi, float3 lightPosOrDir, float3 neighborReceiverPos, inout S sg, inout float3 specularPos, inout uint3 shiftNewtons, bool isPrevFrame = false, bool isTemporalResampling = false, int myMaxIterations = -1)
    {
        shiftNewtons = uint3(0);
        ManifoldVertex si = ManifoldVertex(sd);
        si.dp_du = sdg.dpdu;
        si.dp_dv = sdg.dpdv;
        si.dn_du = sdg.dndu;
        si.dn_dv = sdg.dndv;
        float3 baseEndpoint = si.p;

        // shift the base path
        float3 originalDir = normalize(specularPos - neighborReceiverPos);
        ManifoldVertex offsetPath[maxBounces];
        uint seedMaterialID[maxBounces] = {};
        uint bounces;
        float3 initDir = float3(0.f);
        int2 success_counter = samplePath<S, directional>( baseEndpoint, lightPosOrDir, sg, true, offsetPath, seedMaterialID, bounces, specularPos, initDir, -1, -1, -1, true, float2(0.f), float2(1.f), isPrevFrame, myMaxIterations );

        shiftNewtons.x += 1;
        shiftNewtons.z += success_counter.y;
        bool success = success_counter.x == 1;
        if (!success)
        {
            return float3(0.f);
        }
        float3 offsetSpecularPos = offsetPath[0].p;
        float3 offsetDir = normalize(offsetPath[0].p - baseEndpoint);

        float3 specularVal;
        switch (bounces)
        {
        case 1:
            specularVal = evalSpecularContribution<1, directional>(si, lightPosOrDir, offsetPath);
            break;
        case 2:
            specularVal = evalSpecularContribution<2, directional>(si, lightPosOrDir, offsetPath);
            break;
        default:
            specularVal = float3(0.f);
        }
        if (all(specularVal == 0.f))
        {
            return float3(0.f);
        }

        float3 bsdfVal = mi.eval(sd, offsetDir, sg);

        // check the invertibility
        success_counter = samplePath<S, directional>( neighborReceiverPos, lightPosOrDir, sg, true, offsetPath, seedMaterialID, bounces, offsetSpecularPos, initDir, -1, -1, -1, true, float2(0.f), float2(1.f), isPrevFrame, myMaxIterations );
        shiftNewtons.z += success_counter.y;

        success = success_counter.x == 1;
        if (!success)
        {
            return float3(0.f);
        }
        float3 baseDir = normalize(offsetPath[0].p - neighborReceiverPos);
        if (abs(dot(originalDir, baseDir) - 1.f) > uniqueness_threshold)
        {
            return float3(0.f);
        }

        specularPos = offsetSpecularPos;

        // bijective mappings
        shiftNewtons.y += 1;
        return bsdfVal * specularVal;
    }

    uint2 samplePath<S : ISampleGenerator, let directional : bool>(float3 endpoint, float3 lightPosOrDir, inout S sg, bool firstPath, inout ManifoldVertex currentPath[], inout uint seedMaterialID[], inout uint bounces, float3 specularPos, float3 initDir = float3(0.f), const int sourceTileId = -1, const int threshold = -1, const int firstMaterialID = -1, 
    const bool isUVSpace = true, const float2 uvMin = float2(0.f, 0.f), const float2 uvMax = float2(0.f, 0.f), bool isPrevFrame = false, const int myMaxIterations = -1)
    {
        TextureHandle positionMap[maxBounces];
        TextureHandle shadingNormalMap[maxBounces];
        bool successSeed = sampleSeedPath( endpoint, sg, firstPath, currentPath, positionMap, shadingNormalMap, seedMaterialID, bounces, specularPos, initDir, firstMaterialID, isPrevFrame);
        print("Sample seed success: ", successSeed);
        if (!successSeed)
        {
            return uint2(0, 0);
        }
        switch (bounces)
        {
        case 1:
            return newtonSolver<1, directional>( endpoint, lightPosOrDir, currentPath, positionMap, shadingNormalMap, sourceTileId, threshold, isUVSpace, uvMin, uvMax, isPrevFrame, myMaxIterations);
        case 2:
            return newtonSolver<2, directional>( endpoint, lightPosOrDir, currentPath, positionMap, shadingNormalMap, sourceTileId, threshold, isUVSpace, uvMin, uvMax, isPrevFrame, myMaxIterations);
        default:
            return uint2(0, 0);
        }
    }

    bool sampleSeedPath<S : ISampleGenerator>(float3 endpoint, inout S sg, bool firstPath, out ManifoldVertex currentPath[], out TextureHandle positionMap[], out TextureHandle shadingNormalMap[], inout uint seedMaterialID[], out uint bounces, float3 specularPos, float3 initDir = float3(0.f), const int myFirstMaterialID = -1, bool isPrevFrame = false)
    {
        positionMap = {};
        shadingNormalMap = {};
        currentPath = {};
        bounces = 0;
        float3 initialPos;
        float3 initialDir;
        if (any(initDir != 0.f))
        {
            initialDir = initDir;
        }
        else if (any(specularPos != 0.f))
        {
            initialPos = specularPos;
            initialDir = normalize(initialPos - endpoint);
        }
        else
        {
            uint materialIndex = uint(sampleNext1D(sg) * specularShapesCount);
            int materialID = specularMaterialIDs[materialIndex];
            
            bool isUVSpace = isUVSpaceSampling[materialIndex];
            if(isUVSpace)
            {
                let material = gScene.materials.getMaterial(materialID);
                material.getPositionAndNormalMap(positionMap[0], shadingNormalMap[0]);
                SamplerState s = gScene.materials.getDefaultTextureSampler(materialID);
                initialPos = gScene.materials.sampleTexture(positionMap[0], s, sampleNext2D(sg), 0.f).xyz;
                initialDir = normalize(initialPos - endpoint);
            }
            else
            {
                initialDir = sampleDir(endpoint, sampleNext2D(sg), materialIndex);
            }
        }
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        Ray ray = Ray(endpoint, initialDir, rayEpsilon);

        float hitT;
        const HitInfo hit = sceneRayQuery.traceRay(ray, hitT, 0, 0xff, isPrevFrame);
        if (!(hit.isValid() && hit.getType() == HitType::Triangle)) return false;

        const TriangleHit triangleHit = hit.getTriangleHit();
        const uint firstMaterialID = gScene.getMaterialID(triangleHit.instanceID);

        if(myFirstMaterialID != -1 && firstMaterialID != myFirstMaterialID) return false;

        if (!firstPath && seedMaterialID[0] != firstMaterialID) return false;

        ShadingDifferentialGeometry sdg;
        const ShadingData sd = loadShadingData(hit, ray.dir, sdg, isPrevFrame);
        if (!sd.mtl.isCausticBouncer()) return false;


        bounces = sd.mtl.getCausticBounces();
        currentPath[0].setup(sd);
        currentPath[0].dp_du = sdg.dpdu;
        currentPath[0].dp_dv = sdg.dpdv;
        currentPath[0].dn_du = sdg.dndu;
        currentPath[0].dn_dv = sdg.dndv;

        let tmpMaterial = gScene.materials.getMaterial(firstMaterialID);
        tmpMaterial.getPositionAndNormalMap(positionMap[0], shadingNormalMap[0]);

        if (tmpMaterial.useNormalMap())
        {
            uint hints = getMaterialInstanceHints(hit, false);
            ExplicitLodTextureSampler lod = ExplicitLodTextureSampler(0.f);
            tmpMaterial.getNormalAndDerivative(gScene.materials, sd, lod, hints, sdg, currentPath[0].n, currentPath[0].dn_du, currentPath[0].dn_dv);
            if (!sd.frontFacing)
            {
                currentPath[0].n = -currentPath[0].n;
                currentPath[0].dn_du = -currentPath[0].dn_du;
                currentPath[0].dn_dv = -currentPath[0].dn_dv;
            }
        }

        const float eta = currentPath[0].eta;
        float3 wo;
        if (eta == 1.f)
        {
            wo = reflect(ray.dir, currentPath[0].n);
        }
        else
        {
            float cosi = dot(-ray.dir, currentPath[0].n);
            float cost2 = 1.0 - eta * eta * (1.0 - cosi * cosi);
            if (cost2 < 0.f)
            {
                return false;
            }
            wo = refract(ray.dir, currentPath[0].n, eta);
        }
        ray.origin = sd.posW;
        ray.dir = wo;

        for (uint i = 1; i < bounces && i < maxBounces; ++i)
        {
            float hitT;
            const HitInfo hit = sceneRayQuery.traceRay(ray, hitT, 0, 0xff, isPrevFrame);
            if (!hit.isValid())
            {
                return false;
            }
            if (hit.getType() == HitType::Triangle)
            {
                const TriangleHit triangleHit = hit.getTriangleHit();
                const uint tmpMaterialID = gScene.getMaterialID(triangleHit.instanceID);
                if (!firstPath && seedMaterialID[i] != tmpMaterialID)
                {
                    return false;
                }

                ShadingDifferentialGeometry sdg;
                const ShadingData sd = loadShadingData(hit, ray.dir, sdg, isPrevFrame);
                if (!sd.mtl.isCausticBouncer())
                {
                    // TODO: break
                    return false;
                }
                currentPath[i].setup(sd);
                currentPath[i].dp_du = sdg.dpdu;
                currentPath[i].dp_dv = sdg.dpdv;
                currentPath[i].dn_du = sdg.dndu;
                currentPath[i].dn_dv = sdg.dndv;

                let tmpMaterial = gScene.materials.getMaterial(tmpMaterialID);
                tmpMaterial.getPositionAndNormalMap(positionMap[i], shadingNormalMap[i]);

                if (tmpMaterial.useNormalMap())
                {
                    uint hints = getMaterialInstanceHints(hit, false);
                    ExplicitLodTextureSampler lod = ExplicitLodTextureSampler(0.f);
                    tmpMaterial.getNormalAndDerivative(gScene.materials, sd, lod, hints, sdg, currentPath[i].n, currentPath[i].dn_du, currentPath[i].dn_dv);
                    if (!sd.frontFacing)
                    {
                        currentPath[i].n = -currentPath[i].n;
                        currentPath[i].dn_du = -currentPath[i].dn_du;
                        currentPath[i].dn_dv = -currentPath[i].dn_dv;
                    }
                }

                const float eta = currentPath[i].eta;
                float3 wo;
                if (eta == 1.f)
                {
                    wo = reflect(ray.dir, currentPath[i].n);
                }
                else
                {
                    float cosi = dot(-ray.dir, currentPath[i].n);
                    float cost2 = 1.0 - eta * eta * (1.0 - cosi * cosi);
                    if (cost2 < 0.f)
                    {
                        return false;
                    }
                    wo = refract(ray.dir, currentPath[i].n, eta);
                }
                ray.origin = sd.posW;
                ray.dir = wo;
            }
        }

        if (firstPath)
        {
            for (uint i = 0; i < bounces && i < maxBounces; ++i)
            {
                seedMaterialID[i] = currentPath[i].materialID;
            }
        }

        return true;
    }

    bool tileDistanceInThreshold(int srcTileId, int dstTileId, int numTilesX, int threshold)
    {
        // u, x -> col
        // v, y -> row
        int srcRow = srcTileId / numTilesX;
        int srcCol = srcTileId % numTilesX;
        int dstRow = dstTileId / numTilesX;
        int dstCol = dstTileId % numTilesX;
        int rowDiff = min(abs(srcRow - dstRow), numTilesX - abs(srcRow - dstRow));
        int colDiff = min(abs(srcCol - dstCol), numTilesX - abs(srcCol - dstCol));
        return (rowDiff <= threshold && colDiff <= threshold);
    }
    int getTileId(float2 uv)
    {
        int tileId = int(uv.y * numTilesX) * numTilesX + int(uv.x * numTilesX);
        return tileId;
    }
    

    uint2 newtonSolver<let bounces : uint, let directional : bool>(float3 endpoint, float3 lightPosOrDir, inout ManifoldVertex currentPath[], inout TextureHandle positionMap[], inout TextureHandle shadingNormalMap[], const int sourceTileId = -1, const int threshold = -1, const bool uvSpace = true, const float2 uvMin = float2(0.f, 0.f), const float2 uvMax = float2(0.f, 0.f), const bool isPrevFrame = false, const int myMaxIterations = -1)
    {
        bool success = false;
        uint iterations = 0;
        float beta = 1.f;

        bool needsStepUpdate = true;
        while (iterations < maxIterations)
        {
            if(myMaxIterations != -1 && iterations >= myMaxIterations)
            {
                break;
            }
            bool stepSuccess = true;
            if (needsStepUpdate)
            {
                stepSuccess = computeStepAnglediff<directional>(endpoint, lightPosOrDir, bounces, currentPath);
            }
            if (!stepSuccess)
            {
                break;
            }
            // Check for success
            bool converged = true;
            for (uint i = 0; i < bounces && i < maxBounces; ++i)
            {
                if (length(currentPath[i].C) > solverThreshold)
                {
                    converged = false;
                }
            }
            float2 uv = currentPath[0].uv;
            if (!uvSpace)
            {
                float3 dir = normalize(currentPath[0].p - endpoint);
                float2 uvDir = dirToCanonical(dir);
                uv = uvFromDirSpaceToBound(uvDir, uvMin, uvMax);
            }
            if (sourceTileId != -1 && threshold != -1 && !tileDistanceInThreshold(sourceTileId, getTileId(uv), numTilesX, threshold))
            {
                break;
            }
            if (converged)
            {
                success = true;
                break;
            }
            // Make a proposal
            bool walkSuccess = true;
            ManifoldVertex proposedPath[maxBounces];
            {
                float3 p_prop = currentPath[0].p - beta * (currentPath[0].dp_du * currentPath[0].dx[0] + currentPath[0].dp_dv * currentPath[0].dx[1]);
                walkSuccess = reproject(endpoint, p_prop, bounces, currentPath, proposedPath, isPrevFrame);
            }
            if (walkSuccess)
            {
                beta = min(1.f, 2.f * beta);
                for (uint i = 0; i < bounces && i < maxBounces; ++i)
                {
                    currentPath[i].p = proposedPath[i].p;
                    currentPath[i].n = proposedPath[i].n;
                    currentPath[i].gn = proposedPath[i].gn;
                    currentPath[i].dp_du = proposedPath[i].dp_du;
                    currentPath[i].dp_dv = proposedPath[i].dp_dv;
                    currentPath[i].dn_du = proposedPath[i].dn_du;
                    currentPath[i].dn_dv = proposedPath[i].dn_dv;
                    currentPath[i].uv = proposedPath[i].uv;
                }
                needsStepUpdate = true;
            }
            else
            {
                beta *= 0.5f;
                needsStepUpdate = false;
            }
            iterations++;
        }
        if (!success) {
            return uint2(0, iterations);
        }

        /* In the refraction case, the half-vector formulation of Manifold
           walks will often converge to invalid solutions that are actually
           reflections. Here we need to reject those. */
        for (uint i = 0; i < bounces && i < maxBounces; ++i)
        {
            float3 xPrev = (i == 0) ? endpoint : currentPath[i - 1].p;
            float3 xNext = (i == bounces - 1) ? lightPosOrDir : currentPath[i + 1].p;
            float3 xCur = currentPath[i].p;
            bool at_endpoint_with_fixed_direction = (i == (bounces - 1) && directional);
            float3 wi = normalize(xPrev - xCur);
            float3 wo = at_endpoint_with_fixed_direction ? lightPosOrDir : normalize(xNext - xCur);

            float cos_theta_i = dot(currentPath[i].n, wi),
                  cos_theta_o = dot(currentPath[i].n, wo);
            bool refraction = cos_theta_i * cos_theta_o < 0.f,
                 reflection = !refraction;
            if ((currentPath[i].eta == 1.f && !reflection) ||
                (currentPath[i].eta != 1.f && !refraction)) {
                return uint2(0, iterations);
            }
        }
        return uint2(1, iterations);
    }

    bool reproject(float3 endpoint, float3 firstSpecularPos, uint bounces, inout ManifoldVertex currentPath[], out ManifoldVertex proposedPath[], bool isPrevFrame = false)
    {
        proposedPath = {};
        float3 wo = normalize(firstSpecularPos - endpoint);
        Ray ray = Ray(endpoint, wo, rayEpsilon);
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        for (uint i = 0; i < bounces && i < maxBounces; ++i)
        {
            float hitT;
            const HitInfo hit = sceneRayQuery.traceRay(ray, hitT, 0, 0xff, isPrevFrame);
            if (!hit.isValid() || hit.getType() != HitType::Triangle)
            {
                return false;
            }
            const TriangleHit triangleHit = hit.getTriangleHit();
            const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
            if (materialID != currentPath[i].materialID)
            {
                return false;
            }
            ShadingDifferentialGeometry sdg;
            ShadingData sd = loadShadingData(hit, ray.dir, sdg, isPrevFrame);
            proposedPath[i].setup(sd);
            proposedPath[i].dp_du = sdg.dpdu;
            proposedPath[i].dp_dv = sdg.dpdv;
            proposedPath[i].dn_du = sdg.dndu;
            proposedPath[i].dn_dv = sdg.dndv;

            let material = gScene.materials.getMaterial(materialID);
            if (material.useNormalMap())
            {
                uint hints = getMaterialInstanceHints(hit, false);
                ExplicitLodTextureSampler lod = ExplicitLodTextureSampler(0.f);
                material.getNormalAndDerivative(gScene.materials, sd, lod, hints, sdg, proposedPath[i].n, proposedPath[i].dn_du, proposedPath[i].dn_dv);

                if (!sd.frontFacing)
                {
                    // revert normal
                    proposedPath[i].n = -proposedPath[i].n;
                    proposedPath[i].dn_du = -proposedPath[i].dn_du;
                    proposedPath[i].dn_dv = -proposedPath[i].dn_dv;
                }
            }

            float3 wo;
            const float eta = proposedPath[i].eta;
            if (eta == 1.f)
            {
                wo = reflect(ray.dir, proposedPath[i].n);
            }
            else
            {
                float cosi = dot(-ray.dir, proposedPath[i].n);
                float cost2 = 1.0 - eta * eta * (1.0 - cosi * cosi);
                if (cost2 <= 0.f)
                {
                    return false;
                }
                wo = refract(ray.dir, proposedPath[i].n, eta);
            }
            ray.origin = sd.posW;
            ray.dir = wo;
        }
        return true;
    }

    void getInteraction(float2 uv_prop[], uint bounces, inout ManifoldVertex currentPath[], inout TextureHandle positionMap[], inout TextureHandle shadingNormalMap[])
    {
        for (uint i = 0; i < bounces && i < maxBounces; ++i)
        {
            SamplerState s = gScene.materials.getDefaultTextureSampler(currentPath[i].materialID);
            float3 p = gScene.materials.sampleTexture(positionMap[i], s, uv_prop[i], 0.f).xyz;
            float3 shading_normal = gScene.materials.sampleTexture(shadingNormalMap[i], s, uv_prop[i], 0.f).xyz;
            //float3 face_normal = gScene.materials.sampleTexture(faceNormalMap[i], s, uv_prop[i], 0.f).xyz;

            float3 dpdu, dpdv, dndu, dndv;
            evalDudv(positionMap[i], uv_prop[i], dpdu, dpdv);
            evalDudv(shadingNormalMap[i], uv_prop[i], dndu, dndv);

            currentPath[i].n = shading_normal;
            currentPath[i].p = p;
            currentPath[i].uv = uv_prop[i];
            currentPath[i].dp_du = dpdu;
            currentPath[i].dp_dv = dpdv;
            currentPath[i].dn_du = dndu;
            currentPath[i].dn_dv = dndv;
        }
    }

    float3 evalSpecularContribution<let bounces : uint, let directional : bool>(const ManifoldVertex si, float3 lightPosOrDir, inout ManifoldVertex currentPath[], bool isPrevFrame = false)
    {
        ManifoldVertex vy;
        bool sucess_e = emitterInteractionToVertex<directional>(lightPosOrDir, currentPath[bounces - 1].p, vy, isPrevFrame);
        if (!sucess_e) {
            print("Failed to convert emitter interaction to vertex.", 1);
            return float3(0.f);
        }

        vy.makeOrthonormal();
        for (uint k = 0; k < bounces; ++k) {
            currentPath[k].makeOrthonormal();
        }

        // Shading point to ManifoldVertex
        ManifoldVertex vx = si;
        vx.makeOrthonormal();

        float3 path_throughput = float3(1.f);
        path_throughput *= specularReflectance<directional>(si, lightPosOrDir, bounces, currentPath, isPrevFrame);
        path_throughput *= geometricTerm(vx, vy, bounces, currentPath);
        return path_throughput;
    }

    float3 specularReflectance<let directional : bool>(const ManifoldVertex si, float3 lightPosOrDir, uint bounces, inout ManifoldVertex currentPath[], bool isPrevFrame = false)
    {
        float3 x0 = si.p, x1 = currentPath[0].p;
        float3 wo = normalize(x1 - x0);
        Ray ray = Ray(x0, wo, rayEpsilon);

        float3 bsdfVal = float3(1.f);
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        for (uint i = 0; i < bounces && i < maxBounces; ++i)
        {
            float hitT;
            const HitInfo hit = sceneRayQuery.traceRay(ray, hitT, 0, 0xff, isPrevFrame);
            if (!hit.isValid())
            {
                return float3(0.f);
            }
            if (hit.getType() == HitType::Triangle)
            {
                ShadingDifferentialGeometry sdg;
                ShadingData sd = loadShadingData(hit, ray.dir, isPrevFrame);
                if (!sd.mtl.isCausticBouncer())
                {
                    return float3(0.f);
                }
                float3 wi = -ray.dir;
                float3 wo;
                if (i + 1 < bounces)
                {
                    wo = normalize(currentPath[i + 1].p - sd.posW);
                }
                else
                {
                    if (directional)
                    {
                        wo = lightPosOrDir;
                    }
                    else
                    {
                        wo = normalize(lightPosOrDir - sd.posW);
                    }
                }
                float16_t IoR = sd.mtl.getIoR();
                float eta = currentPath[i].eta;
                float cosTheta = dot(currentPath[i].n, -ray.dir);
                if (eta != 1.f)
                {
                    float F = evalFresnelDielectric(eta, cosTheta);
                    float f = 1 - F;
                    f *= sqrt(eta);
                    bsdfVal *= f;
                }
                else
                {
                    let material = gScene.materials.getMaterial(currentPath[i].materialID);
                    float3 albedo = material.getSpecularReflectionAlbedo();
                    float3 F = evalFresnelSchlick(albedo, 1.f, cosTheta);
                    bsdfVal *= F;
                }

                x1 = ray.origin = sd.posW;
                ray.dir = wo;
            }
        }

        // Do one more ray-trace towards light source to check for visibility there
        // TODO: support analytic lights
        //if (!ei.is_directional()) {
            ray.tMax = length(lightPosOrDir - x1) * (1.f - rayEpsilon);
        //}
        if(directional)
        {
            ray.tMax = 1e8f;
        }
        ray.tMin = 0.001f;
        bool passVisibility = sceneRayQuery.traceVisibilityRay(ray, 0, 0xff, isPrevFrame);
        // print("Visibility: ", passVisibility);
        if (!passVisibility) {
            return float3(0.f);
        }
        // return 1.f;
        // print("BSDF Val: ", bsdfVal);

        return bsdfVal;
    }

    float geometricTerm(const ManifoldVertex vx, const ManifoldVertex vy, uint bounces, inout ManifoldVertex proposedPath[])
    {
        // First assemble full path, including all endpoints (use proposedPath as buffer here)
        // proposedPath.clear();
        int k = 0;
        if (vy.fixedDirection)
        {
            {
                // proposedPath[bounces] = vy;
                // // reverse proposed path (without currentPath) by using swap
                // for (int i = 0, j = bounces; i < j; i++, j--)
                // {
                //     ManifoldVertex tmp = proposedPath[i];
                //     proposedPath[i] = proposedPath[j];
                //     proposedPath[j] = tmp;
                // }
                // proposedPath[bounces + 1] = vx;
                // k = bounces + 2;

                // In general case, we need to reverse the whole path using a loop.
                // For we are only handling the pool scene, which only has 1 bounce
                k = 3;
                proposedPath[1] = proposedPath[0];
                proposedPath[0] = vy;
                proposedPath[2] = vx;
            }
        }
        else 
        {
            proposedPath[bounces] = vy;
            k = bounces + 1;
        }
        for (int i = 0; i + 1 < k; ++i) {
            proposedPath[i].dC_dx_prev = float2x2(0.f);
            proposedPath[i].dC_dx_cur = float2x2(0.f);
            proposedPath[i].dC_dx_next = float2x2(0.f);

            float3 x_cur = proposedPath[i].p;
            float3 x_next = proposedPath[i + 1].p;

            float3 wo = x_next - x_cur;
            float ilo = length(wo);
            if (ilo < 1e-3f) {
                return 0.f;
            }
            ilo = rcp(ilo);
            wo *= ilo;

            if (proposedPath[i].fixedDirection) {
               // Derivative of directional constraint w.r.t. x_{i}
               float3 dc_du_cur = -ilo * (proposedPath[i].dp_du - wo * dot(wo, proposedPath[i].dp_du)),
                      dc_dv_cur = -ilo * (proposedPath[i].dp_dv - wo * dot(wo, proposedPath[i].dp_dv));
               proposedPath[i].dC_dx_cur = float2x2(
                   dot(dc_du_cur, proposedPath[i].dp_du), dot(dc_dv_cur, proposedPath[i].dp_du),
                   dot(dc_du_cur, proposedPath[i].dp_dv), dot(dc_dv_cur, proposedPath[i].dp_dv)
               );

               // Derivative of directional constraint w.r.t. x_{i+1}
               float3 dc_du_next = ilo * (proposedPath[i + 1].dp_du - wo * dot(wo, proposedPath[i + 1].dp_du)),
                      dc_dv_next = ilo * (proposedPath[i + 1].dp_dv - wo * dot(wo, proposedPath[i + 1].dp_dv));
               proposedPath[i].dC_dx_next = float2x2(
                   dot(dc_du_next, proposedPath[i].dp_du), dot(dc_dv_next, proposedPath[i].dp_du),
                   dot(dc_du_next, proposedPath[i].dp_dv), dot(dc_dv_next, proposedPath[i].dp_dv)
               );
               continue;
            }

            float3 x_prev = (i == 0) ? vx.p : proposedPath[i - 1].p; // Note that we only end up here for positionally fixed endpoints, thus x is not part of the path array directly.

            float3 wi = x_prev - x_cur;
            float ili = length(wi);
            if (ili < 1e-3f) {
                return 0.f;
            }
            ili = rcp(ili);
            wi *= ili;

            // Setup generalized half-vector
            // for area
            float eta = rcp(proposedPath[i].eta);

            if (dot(wi, proposedPath[i].n) < 0.f) {
                // this may happen when we reverse the path for directional light
                eta = rcp(eta);
            }

            float3 h = wi + eta * wo;
            float ilh = rcp(length(h));
            h *= ilh;

            ilo *= eta * ilh;
            ili *= ilh;

            // Local shading tangent frame
            float dot_dpdu_n = dot(proposedPath[i].dp_du, proposedPath[i].n),
                  dot_dpdv_n = dot(proposedPath[i].dp_dv, proposedPath[i].n);
            float3 s = proposedPath[i].dp_du - dot_dpdu_n * proposedPath[i].n,
                   t = proposedPath[i].dp_dv - dot_dpdv_n * proposedPath[i].n;

            float3 dh_du, dh_dv;

            // Derivative of specular constraint w.r.t. x_{i-1}
            if (i > 0) {
                dh_du = ili * (proposedPath[i - 1].dp_du - wi * dot(wi, proposedPath[i - 1].dp_du));
                dh_dv = ili * (proposedPath[i - 1].dp_dv - wi * dot(wi, proposedPath[i - 1].dp_dv));
                dh_du -= h * dot(dh_du, h);
                dh_dv -= h * dot(dh_dv, h);

                proposedPath[i].dC_dx_prev = float2x2(
                    dot(s, dh_du), dot(s, dh_dv),
                    dot(t, dh_du), dot(t, dh_dv)
                );
            }

            // Derivative of specular constraint w.r.t. x_{i}
            dh_du = -proposedPath[i].dp_du * (ili + ilo) + wi * (dot(wi, proposedPath[i].dp_du) * ili)
                                                         + wo * (dot(wo, proposedPath[i].dp_du) * ilo);
            dh_dv = -proposedPath[i].dp_dv * (ili + ilo) + wi * (dot(wi, proposedPath[i].dp_dv) * ili)
                                                         + wo * (dot(wo, proposedPath[i].dp_dv) * ilo);
            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);

            float dot_h_n = dot(h, proposedPath[i].n),
                  dot_h_dndu = dot(h, proposedPath[i].dn_du),
                  dot_h_dndv = dot(h, proposedPath[i].dn_dv);

            proposedPath[i].dC_dx_cur = float2x2(
                dot(dh_du, s) - dot(proposedPath[i].dp_du, proposedPath[i].dn_du) * dot_h_n - dot_dpdu_n * dot_h_dndu,
                dot(dh_dv, s) - dot(proposedPath[i].dp_du, proposedPath[i].dn_dv) * dot_h_n - dot_dpdu_n * dot_h_dndv,
                dot(dh_du, t) - dot(proposedPath[i].dp_dv, proposedPath[i].dn_du) * dot_h_n - dot_dpdv_n * dot_h_dndu,
                dot(dh_dv, t) - dot(proposedPath[i].dp_dv, proposedPath[i].dn_dv) * dot_h_n - dot_dpdv_n * dot_h_dndv
            );

            // Derivative of specular constraint w.r.t. x_{i+1}
            dh_du = ilo * (proposedPath[i + 1].dp_du - wo * dot(wo, proposedPath[i + 1].dp_du));
            dh_dv = ilo * (proposedPath[i + 1].dp_dv - wo * dot(wo, proposedPath[i + 1].dp_dv));
            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);

            proposedPath[i].dC_dx_next = float2x2(
                dot(s, dh_du), dot(s, dh_dv),
                dot(t, dh_du), dot(t, dh_dv)
            );
        }
        if (vy.fixedDirection) {
            float G = invertTridiagonalGeo(proposedPath, k);
        //    // Cancel out cosine term that will be added during lighting integral in integrator (in BRDF)
           if (k >= 2)
           {
            float3 d = normalize(proposedPath[k - 1].p - proposedPath[k - 2].p);
            G /= abs(dot(d, proposedPath[k - 1].n));
           }
            G = min(G, 10.f);
           return G;
        } else {
            float dx1_dxend = invertTridiagonalGeo(proposedPath, k);
            /* Unfortunately, these geometric terms can be unstable, so to avoid
               severe variance we can clamp here. */
            // dx1_dxend = min(dx1_dxend, float(2.f));
            float3 d = vx.p - proposedPath[0].p;
            float inv_r2 = rcp(dot(d, d));
            d *= sqrt(inv_r2);
            float dw0_dx1 = abs(dot(d, proposedPath[0].n)) * inv_r2;
            float G = dw0_dx1 * dx1_dxend;
            return G;
        }
    }

    // TODO: we can use slang's autodiff to calculate derivatives
    bool computeStepAnglediff<let directional : bool>(float3 x0, float3 lightPosOrDir, uint bounces, inout ManifoldVertex currentPath[])
    {
        bool success = true;
        for (uint i = 0; i < bounces; ++i) {
            float3 x_prev = (i == 0) ? x0 : currentPath[i - 1].p;
            float3 x_next = (i == bounces - 1) ? lightPosOrDir : currentPath[i + 1].p;
            float3 x_cur = currentPath[i].p;

            bool at_endpoint_with_fixed_direction = (i == (bounces - 1) && directional);
            // Setup wi / wo
            float3 wo;
            if (at_endpoint_with_fixed_direction) {
                wo = lightPosOrDir;
            } else {
                // Standard case for fixed emitter position
                wo = x_next - x_cur;
            }
            float ilo = length(wo);
            if (ilo < 1e-3f) {
                return false;
            }
            ilo = rcp(ilo);
            wo *= ilo;

            float3 dwo_du_cur, dwo_dv_cur;
            if (at_endpoint_with_fixed_direction) {
                dwo_du_cur = float3(0.f);
                dwo_dv_cur = float3(0.f);
            } else {
                // Standard case for fixed emitter position
                dwo_du_cur = -ilo * (currentPath[i].dp_du - wo * dot(wo, currentPath[i].dp_du));
                dwo_dv_cur = -ilo * (currentPath[i].dp_dv - wo * dot(wo, currentPath[i].dp_dv));
            }

            float3 wi = x_prev - x_cur;
            float ili = length(wi);
            if (ili < 1e-3f) {
                return false;
            }
            ili = rcp(ili);
            wi *= ili;

            float3 dwi_du_cur = -ili * (currentPath[i].dp_du - wi * dot(wi, currentPath[i].dp_du)),
                   dwi_dv_cur = -ili * (currentPath[i].dp_dv - wi * dot(wi, currentPath[i].dp_dv));

            // Set up constraint function and its derivatives
            bool success_i = false;
            float3 n = currentPath[i].n;
            float3 dn_du = currentPath[i].dn_du;
            float3 dn_dv = currentPath[i].dn_dv;
            float3 wio;
            bool valid_i_refr_i = transform(wi, n, currentPath[i].eta, wio);
            if (valid_i_refr_i) {
                float to, po, tio, pio;
                sphcoords(wo, to, po);
                sphcoords(wio, tio, pio);

                float dt = to - tio,
                      dp = po - pio;
                if (dp < -M_PI) {
                    dp += 2.f * M_PI;
                } else if (dp > M_PI) {
                    dp -= 2.f * M_PI;
                }
                currentPath[i].C = float2(dt, dp);

                float dto_du, dpo_du, dto_dv, dpo_dv;
                float dtio_du, dpio_du, dtio_dv, dpio_dv;

                // Derivative of specular constraint w.r.t. x_{i-1}
                if (i > 0) {
                    float3 dwi_du_prev = ili * (currentPath[i - 1].dp_du - wi * dot(wi, currentPath[i - 1].dp_du)),
                           dwi_dv_prev = ili * (currentPath[i - 1].dp_dv - wi * dot(wi, currentPath[i - 1].dp_dv));
                    
                    float3 dwio_du_prev, dwio_dv_prev;
                    d_transform( wi, dwi_du_prev, dwi_dv_prev, n, float3(0.f), float3(0.f), currentPath[i].eta, dwio_du_prev, dwio_dv_prev ); // Possible optimization: specific implementation here that already knows some of these are 0.
                    d_sphcoords(wio, dwio_du_prev, dwio_dv_prev, dtio_du, dpio_du, dtio_dv, dpio_dv);

                    currentPath[i].dC_dx_prev[0][0] = -dtio_du;
                    currentPath[i].dC_dx_prev[1][0] = -dpio_du;
                    currentPath[i].dC_dx_prev[0][1] = -dtio_dv;
                    currentPath[i].dC_dx_prev[1][1] = -dpio_dv;
                }

                // Derivative of specular constraint w.r.t. x_{i}
                float3 dwio_du_cur, dwio_dv_cur;
                d_transform(wi, dwi_du_cur, dwi_dv_cur, n, dn_du, dn_dv, currentPath[i].eta, dwio_du_cur, dwio_dv_cur);

                d_sphcoords(wo, dwo_du_cur, dwo_dv_cur, dto_du, dpo_du, dto_dv, dpo_dv);
                d_sphcoords(wio, dwio_du_cur, dwio_dv_cur, dtio_du, dpio_du, dtio_dv, dpio_dv);

                currentPath[i].dC_dx_cur[0][0] = dto_du - dtio_du;
                currentPath[i].dC_dx_cur[1][0] = dpo_du - dpio_du;
                currentPath[i].dC_dx_cur[0][1] = dto_dv - dtio_dv;
                currentPath[i].dC_dx_cur[1][1] = dpo_dv - dpio_dv;

                // Derivative of specular constraint w.r.t. x_{i+1}
                if (i < bounces - 1) {
                    float3 dwo_du_next = ilo * (currentPath[i + 1].dp_du - wo * dot(wo, currentPath[i + 1].dp_du)),
                           dwo_dv_next = ilo * (currentPath[i + 1].dp_dv - wo * dot(wo, currentPath[i + 1].dp_dv));

                    d_sphcoords(wo, dwo_du_next, dwo_dv_next, dto_du, dpo_du, dto_dv, dpo_dv);
                    currentPath[i].dC_dx_next[0][0] = dto_du;
                    currentPath[i].dC_dx_next[1][0] = dpo_du;
                    currentPath[i].dC_dx_next[0][1] = dto_dv;
                    currentPath[i].dC_dx_next[1][1] = dpo_dv;
                }

                success_i = true;
            }

            float3 woi;
            bool valid_o_refr_o = transform(wo, n, currentPath[i].eta, woi);
            if (valid_o_refr_o && !success_i) {
                float ti, pi, toi, poi;
                sphcoords(wi, ti, pi);
                sphcoords(woi, toi, poi);

                float dt = ti - toi,
                      dp = pi - poi;
                if (dp < -M_PI) {
                    dp += 2.f * M_PI;
                } else if (dp > M_PI) {
                    dp -= 2.f * M_PI;
                }
                currentPath[i].C = float2(dt, dp);

                float dti_du, dpi_du, dti_dv, dpi_dv;
                float dtoi_du, dpoi_du, dtoi_dv, dpoi_dv;

                // Derivative of specular constraint w.r.t. x_{i-1}
                if (i > 0) {
                    float3 dwi_du_prev = ili * (currentPath[i - 1].dp_du - wi * dot(wi, currentPath[i - 1].dp_du)),
                           dwi_dv_prev = ili * (currentPath[i - 1].dp_dv - wi * dot(wi, currentPath[i - 1].dp_dv));

                    d_sphcoords(wi, dwi_du_prev, dwi_dv_prev, dti_du, dpi_du, dti_dv, dpi_dv);

                    currentPath[i].dC_dx_prev[0][0] = dti_du;
                    currentPath[i].dC_dx_prev[1][0] = dpi_du;
                    currentPath[i].dC_dx_prev[0][1] = dti_dv;
                    currentPath[i].dC_dx_prev[1][1] = dpi_dv;
                }

                // Derivative of specular constraint w.r.t. x_{i}
                float3 dwoi_du_cur, dwoi_dv_cur;
                d_transform(wo, dwo_du_cur, dwo_dv_cur, n, dn_du, dn_dv, currentPath[i].eta, dwoi_du_cur, dwoi_dv_cur);

                d_sphcoords(wi, dwi_du_cur, dwi_dv_cur, dti_du, dpi_du, dti_dv, dpi_dv);
                d_sphcoords(woi, dwoi_du_cur, dwoi_dv_cur, dtoi_du, dpoi_du, dtoi_dv, dpoi_dv);

                currentPath[i].dC_dx_cur[0][0] = dti_du - dtoi_du;
                currentPath[i].dC_dx_cur[1][0] = dpi_du - dpoi_du;
                currentPath[i].dC_dx_cur[0][1] = dti_dv - dtoi_dv;
                currentPath[i].dC_dx_cur[1][1] = dpi_dv - dpoi_dv;

                // Derivative of specular constraint w.r.t. x_{i+1}
                if (i < bounces - 1) {
                    float3 dwo_du_next = ilo * (currentPath[i + 1].dp_du - wo * dot(wo, currentPath[i + 1].dp_du)),
                           dwo_dv_next = ilo * (currentPath[i + 1].dp_dv - wo * dot(wo, currentPath[i + 1].dp_dv));
                    float3 dwoi_du_next, dwoi_dv_next;
                    d_transform( wo, dwo_du_next, dwo_dv_next, n, float3(0.f), float3(0.f), currentPath[i].eta, dwoi_du_next, dwoi_dv_next ); // Possible optimization: specific implementation here that already knows some of these are 0.

                    // std::tie(dti_du, dpi_du, dti_dv, dpi_dv)  = SpecularManifold::d_sphcoords(wi, dwi_du_next, dwi_dv_next);  // = 0
                    d_sphcoords(woi, dwoi_du_next, dwoi_dv_next, dtoi_du, dpoi_du, dtoi_dv, dpoi_dv);

                    currentPath[i].dC_dx_next[0][0] = -dtoi_du;
                    currentPath[i].dC_dx_next[1][0] = -dpoi_du;
                    currentPath[i].dC_dx_next[0][1] = -dtoi_dv;
                    currentPath[i].dC_dx_next[1][1] = -dpoi_dv;
                }

                success_i = true;
            }

            success &= success_i;
        }
        if (!success || !invertTridiagonalStep(bounces, currentPath)) {
            return false;
        }
        return true;
    }

    [Differentiable]
    static void sphcoords(float3 wo, out float theta, out float phi)
    {
        theta = acos(wo[2]);
        phi = atan2(wo[1], wo[0]);
        if (phi < 0.f)
        {
            phi += 2.f * M_PI;
        }
    }

    bool transform(const float3 w, float3 n, float eta, out float3 wo)
    {
        if (eta == 1.f)
        {
            wo = reflect(-w, n);
            return true;
        }
        else
        {
            if (dot(w, n) < 0.f)
            {
                n *= -1.f;
                eta = rcp(eta);
            }
            float cosTheta = dot(w, n);
            float rootTerm = 1.f - eta * eta * (1.f - cosTheta * cosTheta);
            if (rootTerm < 0.f)
            {
                wo = float3(0.f);
                return false;
            }
            else
            {
                wo = -eta * (w - cosTheta * n) - n * sqrt(rootTerm);
                return true;
            }
        }
    }

    void d_transform(const float3 w, const float3 dw_du, const float3 dw_dv, float3 n, float3 dn_du, float3 dn_dv, float eta, out float3 tmp1, out float3 tmp2)
    {
        if (eta == 1.f)
        {
            float dot_w_n = dot(w, n),
                  dot_dwdu_n = dot(dw_du, n),
                  dot_dwdv_n = dot(dw_dv, n),
                  dot_w_dndu = dot(w, dn_du),
                  dot_w_dndv = dot(w, dn_dv);
            // TODO
            tmp1 = 2.f * ((dot_dwdu_n + dot_w_dndu) * n + dot_w_n * dn_du) - dw_du,
            tmp2 = 2.f * ((dot_dwdv_n + dot_w_dndv) * n + dot_w_n * dn_dv) - dw_dv;
        }
        else
        {
            if (dot(w, n) < 0) {
                // Coming from the "inside"
                n *= -1.f;
                dn_du *= -1.f;
                dn_dv *= -1.f;
                eta = rcp(eta);
            }
            float dot_w_n = dot(w, n),
                  dot_dwdu_n = dot(dw_du, n),
                  dot_dwdv_n = dot(dw_dv, n),
                  dot_w_dndu = dot(w, dn_du),
                  dot_w_dndv = dot(w, dn_dv);
            float root = sqrt(1.f - eta * eta * (1.f - dot_w_n * dot_w_n));
            if (root == 0)
            {
                return;
            }

            float3 a_u = -eta * (dw_du - ((dot_dwdu_n + dot_w_dndu) * n + dot_w_n * dn_du)),
                   b1_u = dn_du * root,
                   b2_u = n * rcp(2.f * root) * (-eta * eta * (-2.f * dot_w_n * (dot_dwdu_n + dot_w_dndu))),
                   b_u = -(b1_u + b2_u),
                   a_v = -eta * (dw_dv - ((dot_dwdv_n + dot_w_dndv) * n + dot_w_n * dn_dv)),
                   b1_v = dn_dv * root,
                   b2_v = n * rcp(2.f * root) * (-eta * eta * (-2.f * dot_w_n * (dot_dwdv_n + dot_w_dndv))),
                   b_v = -(b1_v + b2_v);

            tmp1 = a_u + b_u,
            tmp2 = a_v + b_v;
        }
    }

    void d_sphcoords(const float3 w, const float3 dw_du, const float3 dw_dv, out float tmp1, out float tmp2, out float tmp3, out float tmp4)
    {
        float d_acos = -rcp(sqrt(max(1.f - w[2] * w[2], 0.f)));
        float2 d_theta = d_acos * float2(dw_du[2], dw_dv[2]);

        float yx = w[1] / w[0];
        float d_atan = rcp(1 + yx * yx);
        float2 d_phi = d_atan * float2(w[0] * dw_du[1] - w[1] * dw_du[0],
                                       w[0] * dw_dv[1] - w[1] * dw_dv[0]) * rcp(w[0] * w[0]);
        if (w[0] == 0.f) {
            d_phi = float2(0.f);
        }
        tmp1 = d_theta[0], tmp2 = d_phi[0], tmp3 = d_theta[1], tmp4 = d_phi[1];
    }

    bool invertTridiagonalStep(uint bounces, inout ManifoldVertex v[])
    {
        // Solve block tri-diagonal linear system with full RHS vector

        // From "The Natural-Constraint Representation of the Path Space for Efficient Light Transport Simulation"
        // by Kaplanyan et al. 2014 Supplemental material, Figure 2.

        const uint n = bounces;

        v[0].tmp = v[0].dC_dx_prev;
        float2x2 m = v[0].dC_dx_cur;
        if (!inverse(m, v[0].inv_lambda)) return false;

        for (int i = 1; i < n; ++i) {
            v[i].tmp = mul(v[i].dC_dx_prev, v[i - 1].inv_lambda);
            float2x2 m = v[i].dC_dx_cur - mul(v[i].tmp, v[i - 1].dC_dx_next);
            if (!inverse(m, v[i].inv_lambda)) return false;
        }

        v[0].dx = v[0].C;
        for (int i = 1; i < n; ++i) {
            v[i].dx = v[i].C - mul(v[i].tmp, v[i - 1].dx);
        }

        v[n - 1].dx = mul(v[n - 1].inv_lambda, v[n - 1].dx);
        for (int i = n - 2; i >= 0; --i) {
            v[i].dx = mul(v[i].inv_lambda, v[i].dx - mul(v[i].dC_dx_next, v[i + 1].dx));
        }

        return true;
    }

    // TODO: rename
    bool inverse(const float2x2 A, out float2x2 Ainv)
    {
        float det = determinant(A);
        if (abs(det) == 0) {
            return false;
        }
        float invDet = 1.f / det;
        Ainv[0][0] = A[1][1] * invDet;
        Ainv[1][0] = -A[1][0] * invDet;
        Ainv[0][1] = -A[0][1] * invDet;
        Ainv[1][1] = A[0][0] * invDet;
        return true;
    }

    float invertTridiagonalGeo(inout ManifoldVertex v[], int n)
    {
        // Solve block tri-diagonal linear system with RHS vector where only last element in non-zero

        // Procedure as outlined in original "Manifold Exploration" by Jakob and Marschner 2012.
        // Based on the implementation in Mitsuba 0.6: manifold.cpp line 382
        float2x2 Li;
        if (!inverse(v[0].dC_dx_cur, Li)) return 0.f;

        for (int i = 0; i < n - 2; ++i) {
            v[i].tmp = mul(Li, v[i].dC_dx_next);
            float2x2 m = v[i + 1].dC_dx_cur - mul(v[i + 1].dC_dx_prev, v[i].tmp);
            if (!inverse(m, Li)) return 0.f;
        }

        v[n - 2].inv_lambda = -mul(Li, v[n - 2].dC_dx_next);
        for (int i = n - 3; i >= 0; --i) {
            v[i].inv_lambda = -mul(v[i].tmp, v[i + 1].inv_lambda);
        }

        return abs(determinant(-v[0].inv_lambda));
    }

    void evalDudv(TextureHandle handle, float2 uv, out float3 dxdu, out float3 dxdv)
    {
        handle = gScene.materials.getResolvedTextureHandle(handle, uv);
        Texture2D<float4> tex = gScene.materials.materialTextures[handle.getTextureID()];
        uint width, height;
        tex.GetDimensions(width, height);
        uint2 resolution = uint2(width, height);
        int2 uv_i = clamp(int2(floor(uv * resolution)), int2(0, 0), resolution - 1);
        
        int left_x = max(uv_i.x - 1, 0);
        int right_x = min(uv_i.x + 1, resolution.x - 1);
        int delta_x = right_x - left_x;

        float3 u0 = tex.Load(int3(left_x, uv_i.y, 0)).rgb;
        float3 u1 = tex.Load(int3(right_x, uv_i.y, 0)).rgb;
        dxdu = (u1 - u0) * resolution.x / 2;

        int bottom_y = max(uv_i.y - 1, 0);
        int top_y = min(uv_i.y + 1, resolution.y - 1);
        int delta_y = top_y - bottom_y;

        float3 v0 = tex.Load(int3(uv_i.x, bottom_y, 0)).rgb;
        float3 v1 = tex.Load(int3(uv_i.x, top_y, 0)).rgb;
        dxdv = (v1 - v0) * resolution.y / 2;
    }

    bool emitterInteractionToVertex<let directional : bool>(float3 emitterPosOrDir, float3 lastPos, out ManifoldVertex ei, bool isPrevFrame = false)
    {
        ei = {};
        if (directional)
        {
            ei.fixedDirection = true;
            ei.p = lastPos + emitterPosOrDir;
            ei.n = -emitterPosOrDir;
            // make an orthonormal basis around ei.n
            float3 t = cross(ei.n, float3(0.f, 0.f, 1.f));
            float3 s = cross(ei.n, t);
            ei.dp_du = normalize(s);
            ei.dp_dv = normalize(t);
            return true;
        }
        float3 dir = normalize(emitterPosOrDir - lastPos);
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        Ray ray = Ray(lastPos, dir, rayEpsilon);
        float hitT;
        const HitInfo hit = sceneRayQuery.traceRay(ray, hitT, 0, 0xff, isPrevFrame);
        const float epsilon = 0.1f;
        if (!hit.isValid() || abs(hitT - distance(emitterPosOrDir, lastPos)) > epsilon)
        {
            return false;
        }
        if (hit.getType() != HitType::Triangle)
        {
            return false;
        }
        const TriangleHit triangleHit = hit.getTriangleHit();
        ShadingDifferentialGeometry sdg;
        const ShadingData sd = loadShadingData(hit, ray.dir, sdg, isPrevFrame);
        ei.setup(sd);
        ei.dp_du = sdg.dpdu;
        ei.dp_dv = sdg.dpdv;
        ei.dn_du = sdg.dndu;
        ei.dn_dv = sdg.dndv;
        ei.makeOrthonormal();
        return true;
    }
}

ParameterBlock<SMS> gSMS;
