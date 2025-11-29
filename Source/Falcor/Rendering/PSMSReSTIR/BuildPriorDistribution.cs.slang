#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Scene.Scene;
import Scene.RaytracingInline;
import Utils.Debug.PixelDebug;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import LoadShadingData;
import Utils.Color.ColorHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;
import Reservoir;
import SMS;

#ifndef PRIOR_THREAD_BLOCK_SIZE
#define PRIOR_THREAD_BLOCK_SIZE 256
#endif

#ifndef USE_DIRECTIONAL
#define USE_DIRECTIONAL 0
#endif

groupshared uint2 gSolutions[256];
groupshared uint gLightObjectCounter;
groupshared uint gLightAndObjects[256];
groupshared uint gSolutionTiles[256];
groupshared bool gIsSolutionTile[4096];
groupshared uint gPixelIndex[256];
groupshared uint gSelectedPixelIndex;
groupshared float3 gAnchorPos;

struct BuildPrior
{
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const bool kUseDirectionalLight = USE_DIRECTIONAL;
    RWTexture2D<float4> debugOutput;
    uint2 frameDim;
    uint frameIndex;
    StructuredBuffer<ReceiverInfo> receiverInfos;

    bool calculateCounters;
    RWStructuredBuffer<uint> priorCounters;

    int2 imageBlockDim;
    int numTilesX;

    int numThreadsUsed;

    int passId;

    int envMapNumBlockX;
    int envMapNumBlockY;

    RWStructuredBuffer<uint> solutionTiles;

    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
    static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
    // Samplers
    EnvMapSampler envMapSampler;          ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler; ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

    /** Types of samplable lights.
     */
    enum class LightType
    {
        EnvMap,
        Emissive,
        Analytic
    };

    /** Describes a light sample.
     */
    struct LightSample
    {
        float3 Li;      ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
        float pdf;      ///< Pdf with respect to solid angle at the shading point.
        float3 origin;  ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
        float distance; ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
        float3 dir;     ///< Ray direction for visibility evaluation (normalized).
        uint lightType; ///< Light type this sample comes from (LightType casted to uint).

        uint triangleId;     ///< for reconstructing hitInfo for sampled emissive lights
        float2 barycentrics; ///< for reconstructing hitInfo for sampled emissive lights
        float geometryTerm;  ///< for computing jacobian in GRIS
    };

    /** Describes a path vertex.
     */
    struct PathVertex
    {
        uint index;        ///< Vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
        float3 pos;        ///< Vertex position.
        float3 faceNormal; ///< Geometry normal at the vertex (zero if not on a surface).
        bool frontFacing;  ///< True if path vertex is on the front-facing side (if on a surface).

        /** Initializes a path vertex.
            \param[in] index Vertex index.
            \param[in] pos Vertex position.
            \param[in] faceNormal Geometry normal.
            \param[in] frontFacing Front-facing flag.
        */
        __init(uint index, float3 pos, float3 faceNormal = float3(0.f), bool frontFacing = true)
        {
            this.index = index;
            this.pos = pos;
            this.faceNormal = faceNormal;
            this.frontFacing = frontFacing;
        }

        /** Get position with offset applied in direction of the geometry normal to avoid self-intersection
            for visibility rays.
            \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
            \return Returns the offseted position.
        */
        float3 getRayOrigin(float3 rayDir)
        {
            return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0 ? faceNormal : -faceNormal);
        }

        /** Returns the oriented face normal.
            \return Face normal flipped to the same side as the view vector.
        */
        float3 getOrientedFaceNormal()
        {
            return frontFacing ? faceNormal : -faceNormal;
        }
    }

    /** Generates a light sample on the emissive geometry.
        \param[in] vertex Path vertex.
        \param[in] upperHemisphere True if only upper hemisphere should be considered.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEmissiveSample<S : ISampleGenerator>(const PathVertex vertex, const bool upperHemisphere, inout S sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.
        if (!kUseEmissiveLights) return false;

        // sample a triangle
        if (gScene.lightCollection.getActiveTriangleCount() == 0) return false;
        
        float uLight = sampleNext1D(sg);
        uint triangleCount = gScene.lightCollection.getActiveTriangleCount();
        uint idx = min((uint)(uLight * triangleCount), triangleCount - 1);
        uint triangleIndex = gScene.lightCollection.activeTriangles[idx];
        float triangleSelectionPdf = 1.f / (float)triangleCount;
        TriangleLightSample tls;
        // if (!emissiveSampler.sampleLight(vertex.pos, vertex.getOrientedFaceNormal(), upperHemisphere, sg, tls)) return false;
        float2 u = sampleNext2D(sg);
        if (!sampleTriangle(vertex.pos, triangleIndex, u, tls)) return false;
        tls.pdf *= triangleSelectionPdf;
        // Setup returned sample.
        ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
        ls.pdf = tls.pdf;
        // Offset shading and light position to avoid self-intersection.
        float3 lightPosOrDir = computeRayOrigin(tls.posW, tls.normalW);
        ls.origin = vertex.getRayOrigin(lightPosOrDir - vertex.pos);
        float3 toLight = lightPosOrDir - ls.origin;
        ls.distance = length(toLight);
        ls.dir = normalize(toLight);
        // ReSTIR
        ls.barycentrics = tls.uv;
        ls.triangleId = tls.triangleIndex;
        ls.geometryTerm = tls.geometryTerm;

        return any(ls.Li > 0.f);
    }

    /** Return the probabilities for selecting different light types.
        \param[out] p Probabilities.
    */
    void getLightTypeSelectionProbabilities(out float p[3])
    {
        // Set relative probabilities of the different sampling techniques.
        // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
        p[0] = kUseEnvLight ? 1.f : 0.f;
        p[1] = kUseEmissiveLights ? 1.f : 0.f;
        p[2] = kUseAnalyticLights ? 1.f : 0.f;

        // Normalize probabilities. Early out if zero.
        float sum = p[0] + p[1] + p[2];
        if (sum == 0.f) return;

        float invSum = 1.f / sum;
        p[0] *= invSum;
        p[1] *= invSum;
        p[2] *= invSum;
    }
    float getEnvMapSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[0]; }
    float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[1]; }
    float getAnalyicSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[2]; }

    /** Select a light type for sampling.
        \param[out] lightType Selected light type.
        \param[out] pdf Probability for selected type.
        \param[in,out] sg Sample generator.
        \return Return true if selection is valid.
    */
    bool selectLightType(out uint lightType, out float pdf, inout TinyUniformSampleGenerator sg)
    {
        float p[3];
        getLightTypeSelectionProbabilities(p);

        float u = sampleNext1D(sg);

        [unroll]
        for (lightType = 0; lightType < 3; ++lightType)
        {
            if (u < p[lightType])
            {
                pdf = p[lightType];
                return true;
            }
            u -= p[lightType];
        }

        lightType = {};
        pdf = {};

        return false;
    }

    /** Generates a light sample on the environment map.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEnvMapSample(const PathVertex vertex, inout TinyUniformSampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        if (!kUseEnvLight) return false;

        // Sample environment map.
        EnvMapSample lightSample;
        if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

        // Setup returned sample.
        ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
        ls.pdf = lightSample.pdf;
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        ls.distance = 1e30f;
        ls.dir = lightSample.dir;
        ls.geometryTerm = 1.f;

        // turn ls.dir into uv
        float2 uv = ndir_to_oct_equal_area_unorm(ls.dir);
        ls.barycentrics = uv;

        // calculate light id here based on uv
        int lightId = int(uv.x * envMapNumBlockX) + int(uv.y * envMapNumBlockY) * envMapNumBlockX;
        ls.triangleId = lightId;

        return any(ls.Li > 0.f);
    }
    /** Samples a light source in the scene.
        This function first stochastically selects a type of light source to sample,
        and then calls that the sampling function for the chosen light type.
        \param[in] vertex Path vertex.
        \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
        \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateLightSample( const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout TinyUniformSampleGenerator sg, out LightSample ls )
    {
        ls = {};
        uint lightType;

        bool valid = false;
        if (kUseDirectionalLight && kUseEnvLight)
        {
            valid = generateEnvMapSample(vertex, sg, ls);
            lightType = uint(LightType::EnvMap);
        }
        if (!kUseDirectionalLight && kUseEmissiveLights)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
            valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
            lightType = uint(LightType::Emissive);
        }
        if (!valid) return false;

        // Reject samples in non-requested hemispheres.
        float NdotL = dot(vertex.getOrientedFaceNormal(), ls.dir);
        if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta))
            return false;

        // Account for light type selection.
        ls.lightType = lightType;
        return true;
    }

    bool isValidPixel(uint2 pixel) { return pixel.x < frameDim.x && pixel.y < frameDim.y; }
    uint getPixelIndex(uint2 pixel) { return pixel.y * frameDim.x + pixel.x; }
    int getTileId(float2 uv, const int numTilesX)
    {
        // u, x -> col
        // v, y -> row
        int tileRow = int(uv.y * numTilesX);
        int tileCol = int(uv.x * numTilesX);
        return tileRow * numTilesX + tileCol;
    }

    float2 sampleInTile(const int tileId, const int numTilesX, inout TinyUniformSampleGenerator sg)
    {
        // u, x -> col
        // v, y -> row
        int tileRow = tileId / numTilesX;
        int tileCol = tileId % numTilesX;
        // tile size
        float2 tileOffset = float2(tileCol, tileRow) / numTilesX;
        float2 tileSize = 1.f / numTilesX;
        // sample a uv in the tile
        float2 uv = tileOffset + tileSize * sampleNext2D(sg);
        return uv;
    }

    void sampleSolution(const ReceiverInfo receiverInfo, inout TinyUniformSampleGenerator sg, inout int lightTriangleId, inout int specularIndex, inout int solutionTileId, inout float3 val, inout float3 valWithLight)
    {
        if (!receiverInfo.valid) return;
        HitInfo hit = receiverInfo.hit;
        ShadingDifferentialGeometry sdg;
        const ShadingData sd = loadShadingData(receiverInfo.hit, -receiverInfo.V, sdg);
        ManifoldVertex si = ManifoldVertex(sd);
        si.dp_du = sdg.dpdu;
        si.dp_dv = sdg.dpdv;
        si.dn_du = sdg.dndu;
        si.dn_dv = sdg.dndv;
        float3 specularPos = {};
        if (gSMS.specularShapesCount == 0) return;
        // sample light here
        LightSample ls;
        bool validSample = false;
        PathVertex vertex = PathVertex(1, sd.posW, sd.faceN, sd.frontFacing);
        // Sample a light.
        validSample = generateLightSample(vertex, true, true, sg, ls);
        bool directional = kUseDirectionalLight;
        float3 Le = ls.Li * ls.pdf;
        if (all(Le == 0.f))
        {
            return;
        }
        float3 lightPosOrDir = float3(0.f);
        if (directional)
        {
            lightPosOrDir = ls.dir;
        }
        else
        {
            lightPosOrDir = ls.origin + ls.dir * ls.distance;
        }
        float lightPdf = ls.pdf * ls.geometryTerm;
        ManifoldVertex currentPath[maxBounces];
        uint seedMaterialID[maxBounces];
        uint bounces;

        const uint materialIndex = uint(sampleNext1D(sg) * gSMS.specularShapesCount);
        bool isUVSpace = gSMS.isUVSpaceSampling[materialIndex];
        const uint materialID = gSMS.specularMaterialIDs[materialIndex];

        let material = gScene.materials.getMaterial(materialID);
        TextureHandle positionMap, shadingNormalMap;
        material.getPositionAndNormalMap(positionMap, shadingNormalMap);
        SamplerState s = gScene.materials.getDefaultTextureSampler(materialID);

        // sample uv here
        float2 uv = sampleNext2D(sg);
        float3 initPos = float3(0.f);
        float3 initDir = float3(0.f);
        float2 uvMin = float2(0.f);
        float2 uvMax = float2(1.f);
        AABB aabb = gSMS.specularAABBs[materialIndex];
        uint2 success_counter;
        if (directional)
        {
            success_counter = gSMS.samplePath<TinyUniformSampleGenerator, true>(si.p, lightPosOrDir, sg, true, currentPath, seedMaterialID, bounces, initPos);
        }
        else
        {
            success_counter = gSMS.samplePath<TinyUniformSampleGenerator, false>(si.p, lightPosOrDir, sg, true, currentPath, seedMaterialID, bounces, initPos);
        }
        if (calculateCounters)
        {
            InterlockedAdd(priorCounters[0], 1);
            InterlockedAdd(priorCounters[1], success_counter.x);
            InterlockedAdd(priorCounters[2], success_counter.y);
        }
        bool success = success_counter.x == 1;
        if (!success) return;
        float2 solutionUV = float2(0.f);
        float3 solutionDir = normalize(currentPath[0].p - si.p);
        float2 solutionDirUV = gSMS.dirToCanonical(solutionDir);

        float3 thp;
        float3 specularVal;
        switch (bounces)
        {
        case 1:
            if (directional)
                specularVal = gSMS.evalSpecularContribution<1, true>(si, lightPosOrDir, currentPath);
            else
                specularVal = gSMS.evalSpecularContribution<1, false>(si, lightPosOrDir, currentPath);
            break;
        case 2:
            if (directional)
                specularVal = gSMS.evalSpecularContribution<2, true>(si, lightPosOrDir, currentPath);
            else
                specularVal = gSMS.evalSpecularContribution<2, false>(si, lightPosOrDir, currentPath);
            break;
        default:
            specularVal = float3(0.f);
        }
        if (all(specularVal == 0.f) || any(isnan(specularVal))) return;
        lightTriangleId = ls.triangleId;
        for (int i = 0; i < gSMS.specularShapesCount; i++)
        {
            if (gSMS.specularMaterialIDs[i] == seedMaterialID[0])
            {
                specularIndex = i;
                break;
            }
        }
        if (isUVSpace)
        {
            solutionUV = currentPath[0].uv;
        }
        else
        {
            aabb = gSMS.specularAABBs[specularIndex];
            // // get the uv bound
            gSMS.getDirectionSpaceBound(gAnchorPos, aabb, uvMin, uvMax);
            gSMS.loosenBound(uvMin, uvMax, 0.1f);
            solutionUV = gSMS.uvFromDirSpaceToBound(solutionDirUV, uvMin, uvMax);
        }
        solutionTileId = getTileId(solutionUV, numTilesX);

        val = specularVal;
        valWithLight = Le;
    }

    void process(int imageBlockIndex, int threadIndex)
    {
        // get image block offset
        int2 numBlocks = (frameDim + imageBlockDim - 1) / imageBlockDim;
        int2 imageBlockIndex2D = int2(imageBlockIndex % numBlocks.x, imageBlockIndex / numBlocks.x);
        // get anchor pixel
        uint2 anchorPixel = imageBlockIndex2D * imageBlockDim;

        int2 threadIndex2D = int2(threadIndex % 8, threadIndex / 8);
        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(imageBlockIndex2D + threadIndex2D, frameIndex * 234 + passId);

        // for each thread, randomly select a pixel (or maybe using some pattern, for now we use random sampling)
        uint2 pixel = anchorPixel + sampleNext2D(sg) * imageBlockDim;

        // debug
        printSetPixel(pixel);
        print("Building Prior Distribution -------------------------------------", 1);

        // get pixel index
        int pixelIndex = getPixelIndex(pixel);
        ReceiverInfo receiverInfo;
        receiverInfo.valid = false;
        if (isValidPixel(pixel))
        {
            receiverInfo = receiverInfos[pixelIndex];
        }
        gPixelIndex[threadIndex] = -1;
        if (receiverInfo.valid)
        {
            gPixelIndex[threadIndex] = pixelIndex;
        }
        GroupMemoryBarrierWithGroupSync();

        // randomly select a pixel to get anchor position, we use this anchor position to compute the direction space bound for the pixel block
        if (threadIndex == 0)
        {
            gSelectedPixelIndex = -1;
            gAnchorPos = float3(0.f);
            int idx = -1;
            int cnt = 0;
            for (int i = 0; i < PRIOR_THREAD_BLOCK_SIZE; i++)
            {
                if(gPixelIndex[i] != -1)
                {
                    cnt++;
                    if (sampleNext1D(sg) * cnt < 1.f)
                    {
                        gSelectedPixelIndex = gPixelIndex[i];
                    }
                }
            }
            if (gSelectedPixelIndex != -1)
            {
                HitInfo hit = receiverInfos[gSelectedPixelIndex].hit;
                gAnchorPos = loadVertexPosition(hit);
            }  
        }
        GroupMemoryBarrierWithGroupSync();
        if (gSelectedPixelIndex == -1)
        {
            return;
        }
        int lightTriangleId = -1;
        int specularIndex = -1;
        int solutionTileId = -1;
        float3 val = float3(0.0f);
        float3 valWithLight = float3(0.0f);
        if (receiverInfo.valid && threadIndex <= numThreadsUsed)
        {
            // debugOutput[pixel] = float4(0, 1, 0, 1);
            sampleSolution(receiverInfo, sg, lightTriangleId, specularIndex, solutionTileId, val, valWithLight);
        }
        int totalNumTiles = numTilesX * numTilesX;
        gSolutions[threadIndex] = uint2(-1, -1);
        for (int i = 0; i < totalNumTiles; i += PRIOR_THREAD_BLOCK_SIZE)
        {
            if(i + threadIndex < totalNumTiles)
                gIsSolutionTile[i + threadIndex] = false;
        }
        int lightObjectId = lightTriangleId * 256 + specularIndex;
        if (solutionTileId != -1 && lightTriangleId != -1 && specularIndex != -1)
        {
            // assuming there are at most 256 materials
            gSolutions[threadIndex] = uint2(lightObjectId, solutionTileId);
        }
        GroupMemoryBarrierWithGroupSync();

        // each block keeps an anchor pixel, then light-object with tile counts, followed by tile ids
        int offset = imageBlockIndex * (PRIOR_THREAD_BLOCK_SIZE * 3);
        int rightBound = offset + PRIOR_THREAD_BLOCK_SIZE * 3;
        int debugOffset = 0;
        if (threadIndex == 0)
        {
            gLightObjectCounter = 0;
            gLightAndObjects[0] = -1;
            for (int i = 0; i < PRIOR_THREAD_BLOCK_SIZE; i++)
            {
                int2 solution = gSolutions[i].xy;
                bool found = false;
                if (solution.x != -1)
                {
                    for (int j = 0; j < gLightObjectCounter; j++)
                    {
                        if (gLightAndObjects[j] == solution.x)
                        {
                            found = true;
                        }
                    }
                    if (!found)
                    {
                        gLightAndObjects[gLightObjectCounter++] = solution.x;
                    }
                }
            }
            solutionTiles[offset] = gSelectedPixelIndex;
            offset += 1;
            for (int i = 0; i < gLightObjectCounter; i++)
            {
                int tileCnt = 0;
                int lightObjectId = gLightAndObjects[i];

                for (int j = 0; j < PRIOR_THREAD_BLOCK_SIZE; j++)
                {
                    int2 solution = gSolutions[j].xy;
                    int solutionTileId = solution.y;
                    if (gSolutions[j].x == gLightAndObjects[i] && !gIsSolutionTile[solutionTileId])
                    {
                        gSolutionTiles[tileCnt++] = solutionTileId;
                        gIsSolutionTile[solutionTileId] = true;
                    }
                }
                // write to global buffer
                solutionTiles[offset] = lightObjectId * 256 + tileCnt;
                offset += 1;
                for(int j = 0; j < tileCnt; j++)
                {
                    uint solutionTileId = gSolutionTiles[j];
                    solutionTiles[offset + j] = solutionTileId;
                    gIsSolutionTile[solutionTileId] = false;
                }
                offset += tileCnt;
            }
            if(offset < PRIOR_THREAD_BLOCK_SIZE * 3)
            {
                solutionTiles[offset] = -1;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

ConstantBuffer<BuildPrior> gBuildPrior;

// trace secondary hits
[numthreads(PRIOR_THREAD_BLOCK_SIZE, 1)]
void main(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID)
{
    gBuildPrior.process(groupId.x, groupThreadId.x);
}
