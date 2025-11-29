import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import ReSTIRCommon;
import Rendering.PSMSReSTIR.Params;
import Scene.HitInfo;
import LoadShadingData;
import PathState;
import Scene.RaytracingInline;
import Reservoir;
import Utils.Geometry.GeometryHelpers;
import Utils.Math.MathHelpers;
import SMS;
// import Utils.Sampling.UniformSampleGenerator;
import Utils.Sampling.UniformSampleGenerator;
#include "Utils/Math/MathConstants.slangh"
import Utils.Math.MathHelpers;

#ifndef PRIOR_THREAD_BLOCK_SIZE
#define PRIOR_THREAD_BLOCK_SIZE 256
#endif

#ifndef USE_CONSTRAINT
#define USE_CONSTRAINT 0
#endif

// Compile-time config with safe defaults; renderer sets real values via defines
#ifndef USE_TILING
#define USE_TILING 0
#endif
#ifndef USE_PRIOR
#define USE_PRIOR 0
#endif
#ifndef NUM_TILES_X
#define NUM_TILES_X 16
#endif
#ifndef UNIFORM_THRESHOLD
#define UNIFORM_THRESHOLD 1
#endif
#ifndef PRIOR_THRESHOLD
#define PRIOR_THRESHOLD 1
#endif
#ifndef ALPHA
#define ALPHA 0.5
#endif
#ifndef USE_BOUND_PROB
#define USE_BOUND_PROB 0
#endif

#ifndef USE_DIRECTIONAL
#define USE_DIRECTIONAL 0
#endif

#ifndef USE_OURS
#define USE_OURS 1
#endif

#define PRIOR_BUFFER_SIZE (PRIOR_THREAD_BLOCK_SIZE * 3)

groupshared int gCounter;
groupshared uint gPriorBuffer[PRIOR_BUFFER_SIZE];
groupshared int gLightObjectsCounter;
groupshared int2 gLightObjects[PRIOR_BUFFER_SIZE]; // light object id, offset in the prior buffer
groupshared float gLightObjectEnergy[PRIOR_BUFFER_SIZE];
groupshared float totalLightEnergy;
groupshared float3 gAnchorPos;

struct InitialSampling
{
    ReSTIRPathTracerParams params;

    // Samplers
    EnvMapSampler envMapSampler;                    ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler;           ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

    // Inputs
    Texture2D<PackedHitInfo> vbuffer;

    StructuredBuffer<ReceiverInfo> receiverInfos;
    RWTexture2D<float4> debugOutput; ///< Debug output texture.
    StructuredBuffer<int> solutionTiles;
    int2 imageBlockDim;
    int2 frameDim;

    bool calculateCounters;
    RWStructuredBuffer<uint> initialCounters;

    // Outputs
    RWStructuredBuffer<Reservoir> outputReservoirs;

    int envMapNumBlockX;
    int envMapNumBlockY;
    int2 importanceMapDim;
    StructuredBuffer<float> envMapBlockBuffer;

    uint frameIndex;
    // These runtime flags are now compile-time defines (see k* constants below)

    int passId;

    // Render settings that depend on the scene.
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
    static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
    static const bool kUseOurs = USE_OURS != 0;
    static const bool kUseConstraint = USE_CONSTRAINT != 0;
    static const bool kUseTiling = USE_TILING != 0;
    static const bool kUsePrior = USE_PRIOR != 0;
    static const int  kNumTilesX = NUM_TILES_X;
    static const int  kUniformThreshold = UNIFORM_THRESHOLD;
    static const int  kPriorThreshold = PRIOR_THRESHOLD;
    static const float kAlpha = ALPHA;
    static const bool kUseBoundProb = USE_BOUND_PROB != 0;
    static const bool kUseDirectionalLight = USE_DIRECTIONAL != 0;

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
        float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
        float   pdf;        ///< Pdf with respect to solid angle at the shading point.
        float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
        float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
        float3  dir;        ///< Ray direction for visibility evaluation (normalized).
        uint    lightType;  ///< Light type this sample comes from (LightType casted to uint).

        uint    triangleId;     ///< for reconstructing hitInfo for sampled emissive lights
        float2  barycentrics;   ///< for reconstructing hitInfo for sampled emissive lights
        float   geometryTerm;   ///< for computing jacobian in GRIS
    };

    /** Describes a path vertex.
    */
    struct PathVertex
    {
        uint index;         ///< Vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
        float3 pos;         ///< Vertex position.
        float3 faceNormal;  ///< Geometry normal at the vertex (zero if not on a surface).
        bool frontFacing;   ///< True if path vertex is on the front-facing side (if on a surface).

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

    bool mySampleTriangle<S : ISampleGenerator>(ShadingData sd, int triangleIndex, inout S sg, inout LightSample ls)
    {
        TriangleLightSample tls;
        float2 u = sampleNext2D(sg);
        if (!sampleTriangle(sd.posW, triangleIndex, u, tls)) return false;
        // Setup returned sample.
        ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
        ls.pdf = tls.pdf;
        // Offset shading and light position to avoid self-intersection.
        float3 lightPosOrDir = computeRayOrigin(tls.posW, tls.normalW);
        ls.origin = computeRayOrigin(sd.posW, dot(sd.faceN, lightPosOrDir - sd.posW) >= 0 ? sd.faceN : -sd.faceN);
        float3 toLight = lightPosOrDir - ls.origin;
        ls.distance = length(toLight);
        ls.dir = normalize(toLight);
        // ReSTIR
        ls.barycentrics = tls.uv;
        ls.triangleId = tls.triangleIndex;
        ls.geometryTerm = tls.geometryTerm;
        return any(ls.Li > 0.f);
    }

    bool myGenerateEnvLightSample<S : ISampleGenerator>(const PathVertex vertex, inout S sg, out LightSample ls)
    {
        ls = {};
        if (!kUseEnvLight) return false;
        float threadShould = kAlpha;
        int2 blockSize = (importanceMapDim + int2(envMapNumBlockX, envMapNumBlockY) - 1) / int2(envMapNumBlockX, envMapNumBlockY);
        int sz = blockSize.x * blockSize.y;
        if (kUsePrior && gLightObjectsCounter > 0 && sampleNext1D(sg) < threadShould)
        {
            float2 uv = sampleNext2D(sg);
            float pdf = 0;
            float selected_importance = 0;
            int2 selected_texel = int2(0);
            {
                // sample a block
                float blockPdf = 1.f / (envMapNumBlockX * envMapNumBlockY);
                int blockX = int(uv.x * envMapNumBlockX);
                int blockY = int(uv.y * envMapNumBlockY);
                int blockIndex = blockY * envMapNumBlockX + blockX;
                {
                    // instead of doing random sampling, we can importance sample a block based on sampled solutions
                    uint offset = sampleNext1D(sg) * gLightObjectsCounter;
                    offset = min(offset, gLightObjectsCounter - 1);
                    blockIndex = gLightObjects[offset].x / 256;
                    blockX = blockIndex % envMapNumBlockX;
                    blockY = blockIndex / envMapNumBlockX;
                    blockPdf = 1.f / gLightObjectsCounter;
                    float s = sampleNext1D(sg) * totalLightEnergy;
                    float ss = 0;
                    for (int i = 0; i < gLightObjectsCounter; i++)
                    {
                        ss += gLightObjectEnergy[i];
                        if (ss > s)
                        {
                            blockIndex = gLightObjects[i].x / 256;
                            blockX = blockIndex % envMapNumBlockX;
                            blockY = blockIndex / envMapNumBlockX;
                            blockPdf = gLightObjectEnergy[i] / totalLightEnergy;
                            break;
                        }
                    }
                }
                // sample inside the block
                int blockOffset = blockIndex * sz;
                float totalSum = envMapBlockBuffer[blockOffset + sz - 1];
                float s = sampleNext1D(sg) * totalSum;
                // binary search
                int low = 0;
                int high = sz - 1;
                while (low < high)
                {
                    int mid = (low + high) / 2;
                    if (envMapBlockBuffer[blockOffset + mid] < s)
                    {
                        low = mid + 1;
                    }
                    else
                    {
                        high = mid;
                    }
                }
                int selectedTexelIndexInBlock = high;
                int2 selectedTexelInBlock = int2(selectedTexelIndexInBlock % blockSize.x, selectedTexelIndexInBlock / blockSize.x);
                int2 texel = selectedTexelInBlock + int2(blockX, blockY) * blockSize;
                uv = (texel + sampleNext2D(sg)) / importanceMapDim;

                float prev = selectedTexelIndexInBlock == 0 ? 0 : envMapBlockBuffer[blockOffset + selectedTexelIndexInBlock - 1];
                pdf = (envMapBlockBuffer[blockOffset + selectedTexelIndexInBlock] - prev) / totalSum;
                selected_importance = pdf;
                selected_texel = texel;
                pdf *= blockPdf;
                pdf *= M_1_4PI;
                pdf *= importanceMapDim.x * importanceMapDim.y;

                ls.triangleId = blockIndex;
            }
            float3 dir = oct_to_ndir_equal_area_unorm(uv);
            dir = gScene.envMap.toWorld(dir);
            float envSamplerPdf = envMapSampler.evalPdf(dir);
            float real_pdf = (1 - threadShould) * envSamplerPdf + threadShould * pdf;
            float3 Le = envMapSampler.eval(dir);
            ls.Li = real_pdf > 0.f ? Le / real_pdf : float3(0);
            ls.pdf = real_pdf;
            ls.origin = vertex.getRayOrigin(dir);
            ls.distance = 1e30f;
            ls.dir = dir;
            ls.geometryTerm = 1.f;
        }
        else
        {
            // Sample environment map.
            EnvMapSample lightSample;
            if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;
            float pdf = 0;
            float2 uv = ndir_to_oct_equal_area_unorm(lightSample.dir);
            int2 texel = int2(uv * importanceMapDim);
            int2 block = texel / blockSize;
            int blockId = block.y * envMapNumBlockX + block.x;
            int blockOffset = blockId * sz;
            float totalSum = envMapBlockBuffer[blockOffset + sz - 1];
            ls.triangleId = blockId;
            if(kUsePrior){
                bool found = false;
                for(int i = 0; i < gLightObjectsCounter; i++)
                {
                    if (gLightObjects[i].x / 256 == blockId)
                    {
                        pdf = (gLightObjectEnergy[i] / totalLightEnergy);
                        found = true;
                    }
                }
                if (found)
                {
                    int blockOffset = blockId * sz;
                    int2 texelInBlock = texel - block * blockSize;
                    int texelIndexInBlock = texelInBlock.y * blockSize.x + texelInBlock.x;
                    float prev = texelIndexInBlock == 0 ? 0 : envMapBlockBuffer[blockOffset + texelIndexInBlock - 1];
                    float texelImportance = (envMapBlockBuffer[blockOffset + texelIndexInBlock] - prev) / totalSum;
                    pdf *= texelImportance;
                    pdf *= M_1_4PI;
                    pdf *= importanceMapDim.x * importanceMapDim.y;
                }
            }
            float real_pdf = (1 - threadShould) * lightSample.pdf + threadShould * pdf;
            if (!kUsePrior || gLightObjectsCounter == 0)
            {
                real_pdf = lightSample.pdf;
            }
            // Setup returned sample.
            ls.Li = real_pdf > 0.f ? lightSample.Le / real_pdf : float3(0);
            ls.pdf = real_pdf;
            ls.origin = vertex.getRayOrigin(lightSample.dir);
            ls.distance = 1e30f;
            ls.dir = lightSample.dir;
            ls.geometryTerm = 1.f;
        }
        return true;
    }

    bool generateEnvMapSample<S : ISampleGenerator>(const PathVertex vertex, inout S sg, out LightSample ls)
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
    bool selectLightType<S : ISampleGenerator>(out uint lightType, out float pdf, inout S sg)
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
    bool generateLightSample<S : ISampleGenerator>( const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout S sg, out LightSample ls )
    {
        ls = {};
        uint lightType;
        bool valid = false;
        if (kUseDirectionalLight && kUseEnvLight)
        {
            valid = myGenerateEnvLightSample(vertex, sg, ls);
            lightType = (uint)LightType::EnvMap;
        }
        // if (kUseEnvLight && lightType == (uint)LightType::EnvMap) valid = generateEnvMapSample(vertex, sg, ls);
        if (!kUseDirectionalLight && kUseEmissiveLights)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
            valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
            lightType = (uint)LightType::Emissive;
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

    void run(uint2 pixel)
    {
        printSetPixel(pixel);
        logSetPixel(pixel);

        bool insideScreen = isValidScreenRegion(params, pixel);
        if (!insideScreen) return;

        print("InitialSampling for pixel: ", pixel);

        PathState path;
        generatePath(pixel, path);
        while (path.isActive())
        {
            handleHit(path);
            if (path.isActive()) nextVertex(path);
        }

        const uint centralOffset = params.getReservoirOffset(pixel);
        if (path.reservoir.weight > 0.f)
        {
            debugOutput[pixel] = float4(1, 0, 0, 1);
        }
        outputReservoirs[centralOffset] = path.reservoir;
    }
    void generatePath(uint2 pixel, out PathState path)
    {
        path = {};

        PackedHitInfo primaryHitPacked = vbuffer[pixel];
        if (!isValidPackedHitInfo(primaryHitPacked)) return;

        path.setActive();
        path.setHit(unpackHitInfo(primaryHitPacked));
        path.setPathLength(0);
        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim, false);
        path.origin = cameraRay.origin;
        path.dir = cameraRay.dir;
        path.sg = SampleGenerator(pixel, (1 + 1 + 1) * params.seed + (1 + 1 + 0) + frameIndex * 10 + passId);
    }

    void handleHit(inout PathState path)
    {
        const bool isPrimaryHit = path.getPathLength() == 0;
        const HitType hitType = path.hit.getType();
        if (hitType != HitType::Triangle)
        {
            path.terminate();
            return;
        }
        ShadingDifferentialGeometry sdg;
        ShadingData sd = loadShadingData(path.hit, path.dir, sdg);
        // Create texture sampler.
        let lod = createTextureSampler(isPrimaryHit);

        // Create BSDF instance and query lobe types.
        let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
        const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        const uint lobeTypes = mi.getLobeTypes(sd);
        if (sd.mtl.isCausticBouncer())
        {
            if (sd.mtl.getIoR() == 1.f) sd.mtl.setActiveLobes((uint)LobeType::DeltaReflection);
            else sd.mtl.setActiveLobes((uint)LobeType::DeltaTransmission);
        }
        const bool hasDeltaLobes = (lobeTypes & LobeType::Delta) != 0;
        if (hasDeltaLobes && generateScatterRay(sd, mi, path))
        {
            return;
        }
        if (sd.mtl.isCausticReceiver())
        {
            LightSample ls = {};
            bool validSample = false;
            // Setup path vertex.
            PathVertex vertex = PathVertex(path.getPathLength() + 1, sd.posW, sd.faceN, sd.frontFacing);
            // Sample a light.
            validSample = generateLightSample(vertex, true, true, path.sg, ls);
            float3 lightPosOrDir = float3(0.f);
            bool directional = ls.lightType == uint(LightType.EnvMap);
            if (directional)
            {
                lightPosOrDir = ls.dir;
            }
            else
            {
                lightPosOrDir = ls.origin + ls.dir * ls.distance;
            }
            float lightPdf = ls.pdf * ls.geometryTerm;
            float3 Le = ls.Li * ls.pdf;
            uint3 invProbEstimateNewtons = uint3(0);
            float3 specularPos = float3(0.f);
            // return;
            ManifoldVertex si = ManifoldVertex(sd);
            si.dp_du = sdg.dpdu;
            si.dp_dv = sdg.dpdv;
            si.dn_du = sdg.dndu;
            si.dn_dv = sdg.dndv;
            float3 integrand_counter;
            if (directional)
                integrand_counter = gSMS.specularManifoldSampling<SampleGenerator, true>(si, lightPosOrDir, path.sg, specularPos, invProbEstimateNewtons);
            else
                integrand_counter = gSMS.specularManifoldSampling<SampleGenerator, false>(si, lightPosOrDir, path.sg, specularPos, invProbEstimateNewtons);
            float3 wo = normalize(specularPos - si.p);
            float3 bsdfVal = mi.eval(sd, wo, path.sg);
            
            if (calculateCounters)
            {
                InterlockedAdd(initialCounters[0], 1);

                if (invProbEstimateNewtons.z > 0)
                {
                    InterlockedAdd(initialCounters[1], 1);
                    InterlockedAdd(initialCounters[3], invProbEstimateNewtons.x);
                }
                
                InterlockedAdd(initialCounters[2], invProbEstimateNewtons.y);
                InterlockedAdd(initialCounters[4], invProbEstimateNewtons.z);
            }
            float3 integrand = integrand_counter.xyz;
            if (any(integrand > 0.f))
            {
                // path.reservoir.add(integrand, invProbEstimate / lightPdf, specularPos, lightPosOrDir, directional, Le);
                path.reservoir.add( integrand * Le * bsdfVal, invProbEstimateNewtons.x / lightPdf, specularPos, lightPosOrDir, Le, ls.triangleId, ls.barycentrics);
            }
            path.reservoir.smsInfo.hit = path.hit;
            path.reservoir.smsInfo.dir = -sd.V;
            path.reservoir.M = 1.f;
            path.reservoir.pathLength = path.getPathLength();
            path.terminate();
        }
        path.terminate();
    }

    /** Generates a new scatter ray using BSDF importance sampling.
        \param[in] sd Shading data.
        \param[in] mi Material instance at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        BSDFSample result;
        bool valid = mi.sample(sd, path.sg, result, kUseBSDFSampling);
        if (valid)
        {
            bool isReflection = result.isLobe(LobeType::Reflection);
            path.origin = sd.computeRayOrigin(isReflection);
            path.dir = result.wo;
        }

        return valid;
    }

    void nextVertex(inout PathState path)
    {
        Ray ray = path.getScatterRay();
        float hitT;
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        const HitInfo hit = sceneRayQuery.traceRay(ray, hitT);
        if (hit.isValid())
        {
            path.setHit(hit);
            path.incrementPathLength();
        }
        else
        {
            path.terminate();
        }
    }

    ITextureSampler createTextureSampler(bool isPrimaryHit)
    {
        float lod = isPrimaryHit ? 0.0f : params.lodBias;
        return ExplicitLodTextureSampler(lod);
    }

    float2 sampleInTile(const int tileId, const int kNumTilesX, inout UniformSampleGenerator sg)
    {
        // u, x -> col
        // v, y -> row
        int tileRow = tileId / kNumTilesX;
        int tileCol = tileId % kNumTilesX;
        // tile size
        float2 tileOffset = float2(tileCol, tileRow) / kNumTilesX;
        float2 tileSize = 1.f / kNumTilesX;
        // sample a uv in the tile
        float2 uv = tileOffset + tileSize * sampleNext2D(sg);
        return uv;
    }
    int getTileId(float2 uv, const int kNumTilesX)
    {
        // u, x -> col
        // v, y -> row
        int tileRow = int(uv.y * kNumTilesX);
        int tileCol = int(uv.x * kNumTilesX);
        return tileRow * kNumTilesX + tileCol;
    }
    bool tileDistanceInThreshold(int srcTileId, int dstTileId, int kNumTilesX, int threshold)
    {
        // u, x -> col
        // v, y -> row
        int srcRow = srcTileId / kNumTilesX;
        int srcCol = srcTileId % kNumTilesX;
        int dstRow = dstTileId / kNumTilesX;
        int dstCol = dstTileId % kNumTilesX;
        int rowDiff = min(abs(srcRow - dstRow), kNumTilesX - abs(srcRow - dstRow));
        int colDiff = min(abs(srcCol - dstCol), kNumTilesX - abs(srcCol - dstCol));
        return (rowDiff <= threshold && colDiff <= threshold);
    }
    float2 uniformSampleNearbyTiles(const int tileId, const int kNumTilesX, const int threshold, inout UniformSampleGenerator sg)
    {
        // sample in nearby threshold * threshold tiles, we can reach this 
        int numTotalTiles = kNumTilesX * kNumTilesX;
        int tileRow = tileId / kNumTilesX;
        int tileCol = tileId % kNumTilesX;
        int2 offset = floor(sampleNext2D(sg) * (threshold * 2 + 1)) - threshold;
        int n_row = tileRow + offset.y;
        int n_col = tileCol + offset.x;
        if(n_row < 0) n_row += kNumTilesX;
        if(n_row >= kNumTilesX) n_row -= kNumTilesX;
        if(n_col < 0) n_col += kNumTilesX;
        if(n_col >= kNumTilesX) n_col -= kNumTilesX;
        int n_tileId = n_row * kNumTilesX + n_col;
        // sample a uv in the tile
        float2 uv = sampleInTile(n_tileId, kNumTilesX, sg);
        return uv;
    }
    float2 sampleTilesWithPrior(const int kNumTilesX, const int lightObjectIdx, inout UniformSampleGenerator sg, inout bool isSourceImportant)
    {
        // fuse prior distributions with uniform sampling
        if (!kUsePrior || lightObjectIdx == -1|| sampleNext1D(sg) > kAlpha)
        {
            // uniform sampling
            return sampleNext2D(sg);
        }
        // find the offset
        int offsetInBuffer = gLightObjects[lightObjectIdx].y;
        int v = gPriorBuffer[offsetInBuffer];
        int tileCnt = v % 256;
        int tileOffset = int(sampleNext1D(sg) * tileCnt);
        tileOffset = clamp(tileOffset, 0, tileCnt - 1);
        int tileId = gPriorBuffer[offsetInBuffer + 1 + tileOffset];
        isSourceImportant = true;
        return sampleInTile(tileId, kNumTilesX, sg);
    }
    float2 sampleBoundingTiles(const int tileId, inout UniformSampleGenerator sg, int nearbySolTiles[], int solCnt, inout bool isSourceImportant, float subAlpha)
    {
        if (!kUsePrior || solCnt == 0 || sampleNext1D(sg) > subAlpha)
        {
            return uniformSampleNearbyTiles(tileId, kNumTilesX, kUniformThreshold, sg);
        }
        // sample in the nearby tiles
        int offset = int(sampleNext1D(sg) * solCnt);
        int tileId = nearbySolTiles[offset];
        isSourceImportant = true;
        return sampleInTile(tileId, kNumTilesX, sg);
    }
    void sampleLightObject(inout UniformSampleGenerator sg, inout int lightObjectId, inout int lightObjectIdX, inout float pdf)
    {
        uint triangleCount = gScene.lightCollection.getActiveTriangleCount();
        float priorPdf = 0.f;
        if (sampleNext1D(sg) > kAlpha || gLightObjectsCounter == 0)
        {
            // randomly sample a light and object
            float uLight = sampleNext1D(sg);
            uint idx = min((uint)(uLight * triangleCount), triangleCount - 1);
            uint triangleIndex = gScene.lightCollection.activeTriangles[idx];
            float triangleSelectionPdf = 1.f / (float)triangleCount;
            // sample a specular object
            uint materialIndex = uint(sampleNext1D(sg) * gSMS.specularShapesCount);
            // find the light object idx
            lightObjectId = triangleIndex * 256 + materialIndex;
            // lightObjectId = 0;
            for (int i = 0; i < gLightObjectsCounter; i++)
            {
                if (gLightObjects[i].x == lightObjectId)
                {
                    lightObjectIdX = i;
                    priorPdf = 1.f / gLightObjectsCounter;
                    break;
                }
            }
        }
        else
        {
            // sample light objects based on energy
            float s = sampleNext1D(sg) * totalLightEnergy * 0.9999;
            float ss = 0;
            int idx = 0;
            for(int i = 0; i < gLightObjectsCounter; i++)
            {
                ss += gLightObjectEnergy[i];
                if (ss > s)
                {
                    idx = i;
                    break;
                }
            }
            priorPdf = gLightObjectEnergy[idx] / totalLightEnergy;
            lightObjectId = gLightObjects[idx].x;
            lightObjectIdX = idx;
        }
        // calculate selection probability
        float uniformPdf = 1.f / triangleCount / gSMS.specularShapesCount;
        pdf = kUsePrior && gLightObjectsCounter > 0 ? kAlpha * priorPdf + (1 - kAlpha) * uniformPdf : uniformPdf;
    }

    uint invProbEstimation( float3 siPos, float3 lightPosOrDir, float3 direction, inout ManifoldVertex currentPath[], inout uint seedMaterialID[], inout uint bounces, int firstMaterialID, int solutionTileId, inout UniformSampleGenerator sg, int nearbySolutionTiles[], int cnt, int lightObjectIdx, bool isUVSpace, TextureHandle positionMap, inout SamplerState s, bool useBoundProb, float subAlpha, float2 uvMin, float2 uvMax)
    {
        uint invProbEstimate = 1;
        while (invProbEstimate < kMaxBernoulliTrials)
        {
            float2 uv = float2(0.f);
            bool isSourceImportant = false;
            if (kUseBoundProb) uv = sampleBoundingTiles(solutionTileId, sg, nearbySolutionTiles, cnt, isSourceImportant, subAlpha);
            else uv = sampleTilesWithPrior(kNumTilesX, lightObjectIdx, sg, isSourceImportant);
            int sourceTileId = getTileId(uv, kNumTilesX);
            int threshold = isSourceImportant ? kPriorThreshold : kUniformThreshold;
            float3 initPos = float3(0.f);
            float3 initDir = float3(0.f);
            if (isUVSpace)
            {
                initPos = gScene.materials.sampleTexture(positionMap, s, uv, 0.f).xyz;
                uvMin = float2(0.f);
                uvMax = float2(1.f);
            }
            else
            {
                float2 uvInDir = gSMS.uvFromBoundToDirSpace(uv, uvMin, uvMax);
                initDir = gSMS.canonicalToDir(uvInDir);
            }
            uint2 success_counter;
            success_counter = gSMS.samplePath<UniformSampleGenerator, kUseDirectionalLight>( siPos, lightPosOrDir, sg, false, currentPath, seedMaterialID, bounces, initPos, initDir, sourceTileId, kUseConstraint ? threshold : -1, firstMaterialID, isUVSpace, uvMin, uvMax );
            bool successTrial = success_counter.x == 1;
            if (calculateCounters)
            {
                InterlockedAdd(initialCounters[4], success_counter.y);
            }
            float3 directionTrial = normalize(currentPath[0].p - siPos);
            if (successTrial && abs(dot(direction, directionTrial) - 1.f) < uniqueness_threshold)
            {
                break;
            }
            invProbEstimate++;
        }
        if (calculateCounters)
        {
            InterlockedAdd(initialCounters[3], invProbEstimate);
        }
        return invProbEstimate;
    }

    void SMSwithTiling(const ReceiverInfo receiverInfo, inout Reservoir reservoir, inout UniformSampleGenerator sg)
    {
        reservoir.M = 1.f;
        // load shading data
        HitInfo hit = receiverInfo.hit;
        ShadingDifferentialGeometry sdg;
        ShadingData sd = loadShadingData(receiverInfo.hit, -receiverInfo.V, sdg);
        ManifoldVertex si = ManifoldVertex(sd);
        si.dp_du = sdg.dpdu;
        si.dp_dv = sdg.dpdv;
        si.dn_du = sdg.dndu;
        si.dn_dv = sdg.dndv;
        float3 specularPos = {};
        if (gSMS.specularShapesCount == 0) return;
        // we need to importance sample material index and a light here
        int lightObjectId = 0;
        int lightObjectIdx = -1;
        LightSample ls;
        bool validSample = false;
        float lightPdf = 0.f;
        float3 lightPosOrDir = float3(0.f);
        int materialIndex = -1;
        int lightTriangleId = -1;
        float3 Le = float3(0.f);
        if (!kUseDirectionalLight)
        {
            float pdf = 0.f;
            sampleLightObject(sg, lightObjectId, lightObjectIdx, pdf);
            if (pdf == 0.f) return;
            lightTriangleId = lightObjectId / 256;
            materialIndex = lightObjectId % 256;
            validSample = mySampleTriangle(sd, lightTriangleId, sg, ls);
            Le = ls.Li * ls.pdf;
            lightPdf = ls.pdf * ls.geometryTerm;
            lightPdf *= pdf;
            lightPosOrDir = ls.origin + ls.dir * ls.distance;
        }
        else
        {
            PathVertex vertex = PathVertex(1, sd.posW, sd.faceN, sd.frontFacing);
            bool validSample = generateLightSample(vertex, true, true, sg, ls);
            Le = ls.Li * ls.pdf;
            lightPosOrDir = ls.dir;
            lightPdf = ls.pdf * ls.geometryTerm;

            // here we are only handling the pool scene, so we directly set material index to 0
            materialIndex = 0;
            lightObjectId = ls.triangleId * 256 + materialIndex;
            lightObjectIdx = -1;
        }
        if (gLightObjectsCounter > 0)
        {
            for (int i = 0; i < gLightObjectsCounter; i++)
            {
                if (gLightObjects[i].x == lightObjectId)
                {
                    lightObjectIdx = i;
                    break;
                }
            }
        }
        // sample light source here
        int materialID = gSMS.specularMaterialIDs[materialIndex];
        bool isUVSpace = gSMS.isUVSpaceSampling[materialIndex];
        let material = gScene.materials.getMaterial(materialID);
        TextureHandle positionMap, shadingNormalMap;
        material.getPositionAndNormalMap(positionMap, shadingNormalMap);
        SamplerState s = gScene.materials.getDefaultTextureSampler(materialID);

        ManifoldVertex currentPath[maxBounces];
        uint seedMaterialID[maxBounces];
        uint bounces;
        bool isSourceSolutionTile = false;
        float2 uv = sampleTilesWithPrior(kNumTilesX, lightObjectIdx, sg, isSourceSolutionTile);
        int sourceTileId = getTileId(uv, kNumTilesX);
        AABB aabb = gSMS.specularAABBs[materialIndex];
        float2 uvMin = float2(0.f);
        float2 uvMax = float2(1.f);
        float3 initPos = float3(0.f);
        float3 initDir = float3(0.f);
        float2 uvInDir = float2(0.f);
        if (isUVSpace)
        {
            initPos = gScene.materials.sampleTexture(positionMap, s, uv, 0.f).xyz;
        }
        else
        {
            float3 anchorPos = kUsePrior && lightObjectIdx != -1 ? gAnchorPos : si.p;
            gSMS.getDirectionSpaceBound(anchorPos, aabb, uvMin, uvMax);
            if (kUsePrior && lightObjectIdx != -1)
            {
                gSMS.loosenBound(uvMin, uvMax, 0.1f);
            }
            uvInDir = gSMS.uvFromBoundToDirSpace(uv, uvMin, uvMax);
            initDir = gSMS.canonicalToDir(uvInDir);
        }

        int threshold = isSourceSolutionTile ? kPriorThreshold : kUniformThreshold;
        int firstMaterialID = materialID;
        uint2 success_counter;
        success_counter = gSMS.samplePath<UniformSampleGenerator, kUseDirectionalLight>( si.p, lightPosOrDir, sg, true, currentPath, seedMaterialID, bounces, initPos, initDir, sourceTileId, kUseConstraint ? threshold : -1, firstMaterialID, isUVSpace, uvMin, uvMax );
        if (calculateCounters)
        {
            InterlockedAdd(initialCounters[0], 1);
            InterlockedAdd(initialCounters[1], success_counter.x);
            InterlockedAdd(initialCounters[2], success_counter.y);
        }
        bool success = success_counter.x == 1;
        if (!success || seedMaterialID[0] != materialID) return;
        if (calculateCounters)
        {
            InterlockedAdd(initialCounters[4], success_counter.y);
        }
        specularPos = currentPath[0].p;
        float3 direction = normalize(specularPos - si.p);
        // check solution tile id
        float2 solutionUV = float2(0.f);
        if (isUVSpace)
        {
            solutionUV = currentPath[0].uv;
        }
        else
        {
            solutionUV = gSMS.dirToCanonical(direction);
            solutionUV = gSMS.uvFromDirSpaceToBound(solutionUV, uvMin, uvMax);
        }
        int solutionTileId = getTileId(solutionUV, kNumTilesX);
        int cnt = 0;
        int nearbySolutionTiles[9];
        // get important tile cnt
        int tileCnt = 0;
        float priorProbSum = 0.f;
        if (kUsePrior && kUseBoundProb && lightObjectIdx != -1)
        {
            int offsetInBuffer = gLightObjects[lightObjectIdx].y;
            int v = gPriorBuffer[offsetInBuffer];
            tileCnt = v % 256;
            for (int i = 0; i < tileCnt; i++)
            {
                int tileId = gPriorBuffer[offsetInBuffer + 1 + i];
                if (tileId == -1)
                {
                    return;
                }
                if (tileDistanceInThreshold(solutionTileId, tileId, kNumTilesX, kPriorThreshold))
                {
                    nearbySolutionTiles[cnt++] = tileId;
                    priorProbSum += 1.0 / tileCnt;
                }
            }
        }
        int t = kUniformThreshold * 2 + 1;
        float uniformProb = min(1.f, 1.f * t * t / (kNumTilesX * kNumTilesX));
        float priorProb = lightObjectIdx != -1 ? priorProbSum : 0.f;
        
        float probBound = kUsePrior && (lightObjectIdx != -1) ? kAlpha * priorProb + (1 - kAlpha) * uniformProb : uniformProb;
        float subAlpha = kAlpha * priorProb / probBound;
        if (!kUseBoundProb)
        {
            probBound = 1.f;
        }
        float3 thp;
        float3 specularVal;
        switch (bounces)
        {
        case 1:
            specularVal = gSMS.evalSpecularContribution<1, kUseDirectionalLight>(si, lightPosOrDir, currentPath);
            break;
        case 2:
            specularVal = gSMS.evalSpecularContribution<2, kUseDirectionalLight>(si, lightPosOrDir, currentPath);
            break;
        default:
            specularVal = float3(0.f);
        }
        if (all(specularVal == 0.f)) return;

        let hints = getMaterialInstanceHints(hit, true /* primary hit */);
        let lod = createTextureSampler(receiverInfo.length == 0);
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        float3 bsdfVal = mi.eval(sd, direction, sg);
        uint invProbEstimate = invProbEstimation( si.p, lightPosOrDir, direction, currentPath, seedMaterialID, bounces, firstMaterialID, solutionTileId, sg, nearbySolutionTiles, cnt, lightObjectIdx, isUVSpace, positionMap, s, kUseBoundProb, subAlpha, uvMin, uvMax);
        float3 integrand = bsdfVal * specularVal * Le;
        float invProb = invProbEstimate / probBound;
        reservoir.add(integrand, invProb / lightPdf, specularPos, lightPosOrDir, Le, ls.triangleId, ls.barycentrics);
    }
    void runPrior(uint2 pixel, uint2 threadID)
    {
        printSetPixel(pixel);
        logSetPixel(pixel);
        print("Doing Initial Resampling with Prior ----------", 1);
        bool insideScreen = isValidScreenRegion(params, pixel);
        if (!insideScreen) return;
        UniformSampleGenerator sg = UniformSampleGenerator(pixel, frameIndex * 10 + passId);
        Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim, false);
        int pixelIndex = params.getReservoirOffset(pixel);
        ReceiverInfo receiverInfo = receiverInfos[pixelIndex];

        // calculate image block index and fetch prior buffer
        int threadOffset = threadID.x + threadID.y * 16;
        int2 numBlocks = (frameDim + imageBlockDim - 1) / imageBlockDim;
        int2 block2D = pixel / imageBlockDim;
        int numTotalTiles = NUM_TILES_X * NUM_TILES_X;
        int blockIndex = block2D.x + block2D.y * numBlocks.x;
        int imageBlockOffset = blockIndex * PRIOR_BUFFER_SIZE;
        for (int i = 0; i < PRIOR_BUFFER_SIZE; i += 256)
        {
            int index = imageBlockOffset + i + threadOffset;
            if (i + threadOffset < PRIOR_BUFFER_SIZE)
            {
                gPriorBuffer[i + threadOffset] = solutionTiles[index];
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if(threadOffset == 0)
        {
            // decode the buffer to get all light-objects pairs and their corresponding tiles
            totalLightEnergy = 0;
            gLightObjectsCounter = 0;
            if (kUsePrior)
            {
            //     // load anchor pixel
                gAnchorPos = float3(0.f);
                int pixelIndex = gPriorBuffer[0];
                if (pixelIndex != -1)
                {
                    HitInfo hit = receiverInfos[pixelIndex].hit;
                    gAnchorPos = loadVertexPosition(hit);
                }
                int p = 1;
                while (true)
                {
                    if (gPriorBuffer[p] == -1) break;
                    int v = gPriorBuffer[p];
                    int tileCnt = v % 256;
                    int lightObjectId = v / 256;

                    float val = 1.f;
                    if (kUseDirectionalLight)
                    {
                        int lightBlockId = lightObjectId / 256;
                        int2 blockSize = (importanceMapDim + int2(envMapNumBlockX, envMapNumBlockY) - 1) / int2(envMapNumBlockX, envMapNumBlockY);
                        int sz = blockSize.x * blockSize.y;
                        int lightBlockOffset = lightBlockId * sz;
                        val = envMapBlockBuffer[lightBlockOffset + sz - 1];
                    }
                    else
                    {
                        int lightTriangleId = lightObjectId / 256;
                        float val = gScene.lightCollection.fluxData[lightTriangleId].flux;    
                    }
                    gLightObjectEnergy[gLightObjectsCounter] = val;
                    totalLightEnergy += val;
                    gLightObjects[gLightObjectsCounter++] = int2(lightObjectId, p);
                    p += 1 + tileCnt;
                    if (p >= PRIOR_BUFFER_SIZE) break;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        Reservoir reservoir;
        reservoir.init();
        if (receiverInfo.valid)
        {
            SMSwithTiling(receiverInfo, reservoir, sg);
            reservoir.smsInfo.hit = receiverInfo.hit;
            reservoir.smsInfo.dir = -receiverInfo.V;
            reservoir.pathLength = receiverInfo.length;
        }
        outputReservoirs[params.getReservoirOffset(pixel)] = reservoir;
    }

}

cbuffer CB
{
    InitialSampling gInitialSampling;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID, uint3 groupThreadId: SV_GroupThreadID)
{
    if (!gInitialSampling.kUseOurs)
    {
        gInitialSampling.run(dispatchThreadId.xy);
    }
    else
    {
        gInitialSampling.runPrior(dispatchThreadId.xy, groupThreadId.xy);
    }
}
