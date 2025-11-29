#pragma once
#include "Scene/Scene.h"
#include "RenderGraph/RenderPass.h"
#include "SMS.h"
#include "Params.slang"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Rendering/Lights/EmissiveLightSampler.h"
#include "Rendering/Lights/LightBVHSampler.h"

const int maxNumPasses = 4;

using namespace Falcor;

class FALCOR_API PSMSReSTIRPass
{
public:
/** Configuration options, with generally reasonable defaults.
        */
    struct Options
    {
        bool useTemporalResampling = false;
        bool useSpatialResampling = false;
        bool useTiling = false;
        bool usePriorDistribution = false;

        int buildPriorThreadGroupSize = 128;
        int numThreadsUsedForPrior = 128;
        int2 imageBlockDim = int2(16, 16);
        int numTilesX = 16;
        int uniformThreshold = 4;
        int priorThreshold = 1;
        bool useConstraint = false;
        bool useBoundProb = false;
        bool useDirectional = false;
        float alpha = 0.8f;
        int maxBernoulliTrials = 128;

        uint mSpatialNeighborCount = 1;
        float mSpatialGatherRadius = 30.f;
        int reuseMaxIterations = 5;

        float solverThreshold = 1e-4f;

        Options() {}

        template<typename Archive>
        void serialize(Archive& ar)
        {
            ar("useTemporalResampling", useTemporalResampling);
            ar("useSpatialResampling", useSpatialResampling);
            ar("useTiling", useTiling);
            ar("usePriorDistribution", usePriorDistribution);
            ar("buildPriorThreadGroupSize", buildPriorThreadGroupSize);
            ar("numThreadsUsedForPrior", numThreadsUsedForPrior);
            ar("imageBlockDim", imageBlockDim);
            ar("numTilesX", numTilesX);
            ar("uniformThreshold", uniformThreshold);
            ar("priorThreshold", priorThreshold);
            ar("useConstraint", useConstraint);
            ar("useBoundProb", useBoundProb);
            ar("useDirectional", useDirectional);
            ar("alpha", alpha);
            ar("maxBernoulliTrials", maxBernoulliTrials);
            ar("mSpatialNeighborCount", mSpatialNeighborCount);
            ar("mSpatialGatherRadius", mSpatialGatherRadius);
            ar("reuseMaxIterations", reuseMaxIterations);
            ar("solverThreshold", solverThreshold);
        }
    };
    void setOptions(const Options& options);
    const Options& getOptions() { return mOptions; }
    PSMSReSTIRPass(const ref<Scene>& pScene, const Options& options, const DefineList& defines = DefineList());

    void setOwnerDefines(DefineList defines) { mDefines = defines; }

    void beginFrame(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles, bool needRecompile);
    void prepareResources(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles);
    void prepareLighting(RenderContext* pRenderContext);
    void updatePrograms();
    void bindShaderData(const ShaderVar& rootVar) const;
    void update(RenderContext* pRenderContext, const ref<Texture>& pVbuffer, const ref<Texture>& pMotionVectors, const std::unique_ptr<SMS>& pSMS,
        const std::unique_ptr<EmissiveLightSampler>& emissiveSampler, const std::unique_ptr<EnvMapSampler>& envMapSampler);
    void endFrame(RenderContext* pRenderContext);

    bool renderUI(Gui::Widgets& widget);
    void setRecompile(bool recompile) { mRecompile = recompile; }
    void setReSTIRParams(int useFixedSeed, uint fixedSeed,
        float lodBias, float specularRoughnessThreshold, uint2 frameDim, uint2 screenTiles, uint frameCount, uint seed);

    const ref<Texture>& getDebugOutputTexture() const { return mpDebugOutputTexture; }
    const std::unique_ptr<PixelDebug>& getPixelDebug() const { return mpPixelDebug; }

private:
    Options mOptions;
    ref<Scene> mpScene;
    ref<Device> mpDevice;
    DefineList mDefines;

    ref<ComputePass> mpReflectTypes;
    ref<ComputePass> mpInitialSamplingPass;
    ref<ComputePass> mpTemporalResamplingPass;
    ref<ComputePass> mpSpatialResamplingPass;
    ref<ComputePass> mpResolvePass;

    ref<ComputePass> mpTraceReceiverPass;
    ref<ComputePass> mpBuildPriorPass;

    ref<Buffer> mpCurrentReceiverInfo;

    // 0 -> Samples
    // 1 -> Number of solutions
    // 2 -> Total newton iterations
    ref<Buffer> mpPriorCounters;
    
    // 0 -> Samples
    // 1 -> Solutions
    // 2 -> Total newton iterations during sampling
    // 3 -> Total Bernoulli trials 
    // 4 -> Total newton iterations for solutions

    // finally, we want:
    // 1. Sampling success rate
    // 2. Newtons per sample (success + fail)
    // 3. Bernoulli per solution
    // 4. Newtons per solution
    ref<Buffer> mpInitialCounters;

    // 0 -> Shift mappings
    // 1 -> Successful bijective shifts
    // 2 -> Total newton iterations
    // we want:
    // 1. Shift success rate
    // 2. Newton iterations per shift
    ref<Buffer> mpTemporalCounters;

    // 0 -> Shift mappings
    // 1 -> Successful shifts
    // 2 -> Total newton iterations
    ref<Buffer> mpSpatialCounters;

    ReSTIRPathTracerParams mParams;
    ref<Buffer> mpSolutionTiles[maxNumPasses];
    ref<Buffer> mpTemporalReservoirs[maxNumPasses];
    ref<Buffer> mpOutputReservoirs[maxNumPasses];
    ref<Texture> mpNeighborOffsets;

    // used for plotting intermediate data
    ref<Texture> mpDebugOutputTexture;

    ref<Texture> mpFinalThp;

    std::unique_ptr<PixelDebug> mpPixelDebug;

    struct StaticParams
    {
        // EmissiveLightSamplerType emissiveSampler = EmissiveLightSamplerType::LightBVH;
        EmissiveLightSamplerType emissiveSampler = EmissiveLightSamplerType::Uniform;
    };

    StaticParams mStaticParams;
    LightBVHSampler::Options mLightBVHOptions;
    uint2 mFrameDim = uint2(0);
    uint mFrameIndex = 0;
    bool mRecompile = true;

    bool useOurs = true;
    int numPasses = 1;
    bool mCalculateCounters = false;

    int envMapNumBlockX = 8;
    int envMapNumBlockY = 8;
    int2 importanceMapDim = int2(512, 512);
    int prevEnvMapNumBlockX = 0;
    int prevEnvMapNumBlockY = 0;
    ref<Buffer> mpEnvMapBlockBuffer;
    ref<ComputePass> mpWriteToEnvBuffer;

    void traceReceiver(RenderContext* pRenderContext);
    void buildPrior(RenderContext* pRenderContext, const std::unique_ptr<SMS>& pSMS, const std::unique_ptr<EmissiveLightSampler>& emissiveSampler, const std::unique_ptr<EnvMapSampler>& envMapSampler, int passId);
    void initialSampling(RenderContext* pRenderContext, const ref<Texture>& pVbuffer, const std::unique_ptr<SMS>& pSMS, const std::unique_ptr<EmissiveLightSampler>& emissiveSampler, const std::unique_ptr<EnvMapSampler>& envMapSampler, int passId);
    void temporalResampling(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors, const std::unique_ptr<SMS>& pSMS, int passId);
    void spatialResampling(RenderContext* pRenderContext, const std::unique_ptr<SMS>& pSMS, int passId);
    void resolve(RenderContext* pRenderContext);
    void writeToEnvBuffer(RenderContext* pRenderContext);

    ref<Texture> createNeighborOffsetTexture(uint32_t sampleCount);
};
