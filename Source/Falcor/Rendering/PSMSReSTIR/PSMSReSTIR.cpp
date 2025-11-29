#include "PSMSReSTIR.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"

namespace
{
    const std::string kReflectTypesFilename = "Rendering/PSMSReSTIR/ReflectTypes.cs.slang";
    const std::string kInitialSamplingPassFilename = "Rendering/PSMSReSTIR/InitialSampling.cs.slang";
    const std::string kTemporalResamplingPassFilename = "Rendering/PSMSReSTIR/TemporalResampling.cs.slang";
    const std::string kSpatialResamplingPassFilename = "Rendering/PSMSReSTIR/SpatialResampling.cs.slang";
    const std::string kTraceReceiverPassFilename = "Rendering/PSMSReSTIR/TraceReceiver.cs.slang";
    const std::string kBuildPriorPassFilename = "Rendering/PSMSReSTIR/BuildPriorDistribution.cs.slang";
    const std::string kResolvePassFilename = "Rendering/PSMSReSTIR/Resolve.cs.slang";
    const std::string kWriteToEnvBufferFilename = "Rendering/PSMSReSTIR/WriteToEnvBuffer.cs.slang";

    const uint32_t kNeighborOffsetCount = 8192;
    const uint2 kScreenTileDim = {16, 16};
}

PSMSReSTIRPass::PSMSReSTIRPass(const ref<Scene>& pScene, const Options& options, const DefineList& defines)
    : mOptions(options)
    , mpScene(pScene)
    , mpDevice(pScene->getDevice())
    , mDefines(defines)
{
    FALCOR_ASSERT(mpScene);

    // Create neighbor offset texture.
    mpNeighborOffsets = createNeighborOffsetTexture(kNeighborOffsetCount);

    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);
}

void PSMSReSTIRPass::setOptions(const Options& options)
{
    if (std::memcmp(&options, &mOptions, sizeof(Options)) != 0)
    {
        mOptions = options;
        mRecompile = true;
    }
}

void PSMSReSTIRPass::beginFrame(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles, bool needRecompile)
{
    mRecompile |= needRecompile;
    mFrameDim = frameDim;

    prepareLighting(pRenderContext);
    updatePrograms();
    prepareResources(pRenderContext, mFrameDim, screenTiles);
    mpPixelDebug->beginFrame(pRenderContext, mFrameDim);
}

void PSMSReSTIRPass::updatePrograms()
{
    if (!mRecompile) return;

    mFrameIndex = 0;

    DefineList commonDefines;
    commonDefines.add(mpScene->getSceneDefines());
    commonDefines.add(mDefines);
    commonDefines.add("PRIOR_THREAD_BLOCK_SIZE", std::to_string(mOptions.buildPriorThreadGroupSize));
    commonDefines.add("MAX_BERNOULLI_TRIALS", std::to_string(mOptions.maxBernoulliTrials));
    commonDefines.add("SOLVER_THRESHOLD", std::to_string(mOptions.solverThreshold));

    TypeConformanceList typeConformances;
    // Scene-specific configuration.
    typeConformances.add(mpScene->getTypeConformances());

    ProgramDesc baseDesc;
    baseDesc.addShaderModules(mpScene->getShaderModules());
    baseDesc.addTypeConformances(typeConformances);

    if (!mpReflectTypes)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kReflectTypesFilename).csEntry("main");
        mpReflectTypes = ComputePass::create(mpDevice, desc, commonDefines);
    }

    if(!mpTraceReceiverPass)
    {
        ProgramDesc desc = baseDesc;
        DefineList defines = commonDefines;
        desc.addShaderLibrary(kTraceReceiverPassFilename).csEntry("main");
        mpTraceReceiverPass = ComputePass::create(mpDevice, desc, defines);
    }

    // recreate these two passes every recompliation
    if(!mpBuildPriorPass)
    {
        ProgramDesc desc = baseDesc;
        DefineList defines = commonDefines;
        desc.addShaderLibrary(kBuildPriorPassFilename).csEntry("main");
        mpBuildPriorPass = ComputePass::create(mpDevice, desc, defines);
    }
    if (!mpInitialSamplingPass)
    {
        ProgramDesc desc = baseDesc;
        DefineList defines = commonDefines;
        desc.addShaderLibrary(kInitialSamplingPassFilename).csEntry("main");
        mpInitialSamplingPass = ComputePass::create(mpDevice, desc, defines);
    }

    if (!mpTemporalResamplingPass && mOptions.useTemporalResampling)
    {
        ProgramDesc desc = baseDesc;
        DefineList defines = commonDefines;
        desc.addShaderLibrary(kTemporalResamplingPassFilename).csEntry("main");
        mpTemporalResamplingPass = ComputePass::create(mpDevice, desc, defines);
    }

    if (!mpSpatialResamplingPass && mOptions.useSpatialResampling)
    {
        ProgramDesc desc = baseDesc;
        DefineList defines = commonDefines;
        defines.add("NEIGHBOR_OFFSET_COUNT", std::to_string(mpNeighborOffsets->getWidth()));
        desc.addShaderLibrary(kSpatialResamplingPassFilename).csEntry("main");
        mpSpatialResamplingPass = ComputePass::create(mpDevice, desc, defines);
    }
    if (!mpResolvePass)
    {
        ProgramDesc desc = baseDesc;
        DefineList defines = commonDefines;
        desc.addShaderLibrary(kResolvePassFilename).csEntry("main");
        mpResolvePass = ComputePass::create(mpDevice, desc, defines);
    }
    if (!mpWriteToEnvBuffer)
    {
        ProgramDesc desc = baseDesc;
        DefineList defines = commonDefines;
        desc.addShaderLibrary(kWriteToEnvBufferFilename).csEntry("main");
        mpWriteToEnvBuffer = ComputePass::create(mpDevice, desc, defines);
    }

    mRecompile = false;
}

void PSMSReSTIRPass::prepareResources(RenderContext* pRenderContext, const uint2& frameDim, const uint2& screenTiles)
{
    // Create screen sized buffers.
    uint32_t tileCount = screenTiles.x * screenTiles.y;
    const uint32_t elementCount = tileCount * kScreenTileDim.x * kScreenTileDim.y;

    auto reflectVar = mpReflectTypes->getRootVar();
    for(int i = 0; i < maxNumPasses; i++)
    {
        if (!mpOutputReservoirs[i] || mpOutputReservoirs[i]->getElementCount() < elementCount)
        {
            mpOutputReservoirs[i] = mpDevice->createStructuredBuffer(
                reflectVar["reservoirs"],
                elementCount,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr,
                false
            );
        }
        if (!mpTemporalReservoirs[i] || mpTemporalReservoirs[i]->getElementCount() < elementCount)
        {
            mpTemporalReservoirs[i] = mpDevice->createStructuredBuffer(
                reflectVar["reservoirs"],
                elementCount,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr,
                false
            );
        }
        if(!mpSolutionTiles[i] || mpSolutionTiles[i]->getElementCount() < elementCount * 4)
        {
            mpSolutionTiles[i] = mpDevice->createStructuredBuffer(
                sizeof(uint32_t),
                elementCount * 4,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr,
                false
            );
        }
    }

    // create debug output texture
    if (!mpDebugOutputTexture || mpDebugOutputTexture->getWidth() != mFrameDim.x || mpDebugOutputTexture->getHeight() != mFrameDim.y)
    {
        mpDebugOutputTexture = mpDevice->createTexture2D(mFrameDim.x, mFrameDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    }
    if(!mpFinalThp || mpFinalThp->getWidth() != mFrameDim.x || mpFinalThp->getHeight() != mFrameDim.y)
    {
        mpFinalThp = mpDevice->createTexture2D(mFrameDim.x, mFrameDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    }
    if(!mpCurrentReceiverInfo || mpCurrentReceiverInfo->getElementCount() < elementCount)
    {
        mpCurrentReceiverInfo = mpDevice->createStructuredBuffer(
            reflectVar["receiverInfos"],
            elementCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
    }
    if(!mpPriorCounters || mpPriorCounters->getElementCount() < 10)
    {
        mpPriorCounters = mpDevice->createStructuredBuffer(
            sizeof(10),
            elementCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
    }
    if(!mpInitialCounters || mpInitialCounters->getElementCount() < 10)
    {
        mpInitialCounters = mpDevice->createStructuredBuffer(
            sizeof(10),
            elementCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
    }
    if(!mpTemporalCounters || mpTemporalCounters->getElementCount() < 10)
    {
        mpTemporalCounters = mpDevice->createStructuredBuffer(
            sizeof(10),
            elementCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
    }
    if(!mpSpatialCounters || mpSpatialCounters->getElementCount() < 10)
    {
        mpSpatialCounters = mpDevice->createStructuredBuffer(
            sizeof(10),
            elementCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
    }
    uint32_t importanceMapTexels = 1024 * 1024;
    if(!mpEnvMapBlockBuffer || mpEnvMapBlockBuffer->getElementCount() < importanceMapTexels * 2)
    {
        mpEnvMapBlockBuffer = mpDevice->createStructuredBuffer(sizeof(float), importanceMapTexels * 2, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
    }
}

void PSMSReSTIRPass::prepareLighting(RenderContext* pRenderContext)
{
    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getILightCollection(pRenderContext);
    }
}

void PSMSReSTIRPass::bindShaderData(const ShaderVar& rootVar) const
{
    auto var = rootVar["gReSTIR"];
    var["params"].setBlob(mParams);
    var["finalThp"] = mpFinalThp;
}

void PSMSReSTIRPass::traceReceiver(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "Trace Receiver");

    // Bind resources.
    auto rootVar = mpTraceReceiverPass->getRootVar();
    mpScene->bindShaderData(rootVar["gScene"]);
    mpPixelDebug->prepareProgram(mpTraceReceiverPass->getProgram(), rootVar);

    auto var = rootVar["gTraceReceiver"];
    var["receiverInfos"] = mpCurrentReceiverInfo;
    var["maxDepth"] = 4;
    var["frameIndex"] = mFrameIndex;
    var["frameDim"] = mFrameDim;
    var["debugOutput"] = mpDebugOutputTexture;

    // Dispatch.
    mpTraceReceiverPass->execute(pRenderContext, {mFrameDim.x, mFrameDim.y, 1u});
}

void PSMSReSTIRPass::buildPrior(RenderContext* pRenderContext, const std::unique_ptr<SMS>& pSMS, const std::unique_ptr<EmissiveLightSampler>& emissiveSampler, const std::unique_ptr<EnvMapSampler>& envMapSampler, int passId)
{
    FALCOR_PROFILE(pRenderContext, "Build Prior");
    mpBuildPriorPass->addDefine("PRIOR_THREAD_BLOCK_SIZE", std::to_string(mOptions.buildPriorThreadGroupSize));
    mpBuildPriorPass->addDefine("USE_DIRECTIONAL", std::to_string(mOptions.useDirectional));

    // clear solution tiles
    pRenderContext->clearUAV(mpSolutionTiles[passId]->getUAV().get(), uint4(-1));

    // Bind resources.
    auto rootVar = mpBuildPriorPass->getRootVar();
    mpScene->bindShaderData(rootVar["gScene"]);
    mpPixelDebug->prepareProgram(mpBuildPriorPass->getProgram(), rootVar);
    pSMS->bindShaderData(rootVar["gSMS"], mOptions.numTilesX);
    auto var = rootVar["gBuildPrior"];
    var["receiverInfos"] = mpCurrentReceiverInfo;
    var["solutionTiles"] = mpSolutionTiles[passId];
    var["frameIndex"] = mFrameIndex;
    var["frameDim"] = mFrameDim;
    var["debugOutput"] = mpDebugOutputTexture;
    var["numTilesX"] = mOptions.numTilesX;
    var["imageBlockDim"] = mOptions.imageBlockDim;

    var["numThreadsUsed"] = mOptions.numThreadsUsedForPrior;

    var["calculateCounters"] = mCalculateCounters;
    var["priorCounters"] = mpPriorCounters;

    var["envMapNumBlockX"] = envMapNumBlockX;
    var["envMapNumBlockY"] = envMapNumBlockY;

    if (envMapSampler) envMapSampler->bindShaderData(var["envMapSampler"]);
    if (emissiveSampler) emissiveSampler->bindShaderData(var["emissiveSampler"]);

    // calculate num blocks
    uint2 numBlocks = { (mFrameDim.x + mOptions.imageBlockDim.x - 1) / mOptions.imageBlockDim.x, (mFrameDim.y + mOptions.imageBlockDim.y - 1) / mOptions.imageBlockDim.y };
    mpBuildPriorPass->execute(pRenderContext, { numBlocks.x * numBlocks.y * mOptions.buildPriorThreadGroupSize, 1u, 1u });
}

void PSMSReSTIRPass::writeToEnvBuffer(RenderContext* pRenderContext)
{
    if (!mpWriteToEnvBuffer) return;

    pRenderContext->clearUAV(mpEnvMapBlockBuffer->getUAV().get(), uint4(0));

    // Set the input and output resources for the compute pass.
    auto rootVar = mpWriteToEnvBuffer->getRootVar();
    mpScene->bindShaderData(rootVar["gScene"]);

    auto var = rootVar["gWriteToEnvMapBuffer"];

    var["envMapImportanceBuffer"] = mpEnvMapBlockBuffer;
    var["envMapNumBlockX"] = envMapNumBlockX;
    var["envMapNumBlockY"] = envMapNumBlockY;
    var["importanceMapDim"] = importanceMapDim;

    mpWriteToEnvBuffer->execute(pRenderContext, {importanceMapDim, 1});
}

void PSMSReSTIRPass::update(RenderContext* pRenderContext, const ref<Texture>& pVbuffer, const ref<Texture>& pMotionVectors, const std::unique_ptr<SMS>& pSMS,
    const std::unique_ptr<EmissiveLightSampler>& emissiveSampler, const std::unique_ptr<EnvMapSampler>& envMapSampler)
{
    // clear counters
    // create prefix sum on CPU here by ourselves
    if(prevEnvMapNumBlockX != envMapNumBlockX || prevEnvMapNumBlockY != envMapNumBlockY)
    {
        if(mpWriteToEnvBuffer && mpScene->getEnvMap())
        {
            // write to env map buffers
            writeToEnvBuffer(pRenderContext);
            std::vector<float> result = mpEnvMapBlockBuffer->getElements<float>();
            int2 blockSize = (importanceMapDim + int2(envMapNumBlockX, envMapNumBlockY) - 1) / int2(envMapNumBlockX, envMapNumBlockY);
            int numBlocks = envMapNumBlockX * envMapNumBlockY;
            int sz = blockSize.x * blockSize.y;
            for(int i = 0; i < numBlocks; i++)
            {
                int blockOffset = i * sz;
                // do prefix sum by ourselves
                for(int j = blockOffset; j < blockOffset + sz; j++)
                {
                    // exclusive prefix sum
                    result[j] += (j > blockOffset) ? result[j - 1] : 0;
                }
                ref<Buffer> pTemp = mpDevice->createBuffer(sz * sizeof(float), ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, result.data() + blockOffset);
                pRenderContext->copyBufferRegion(mpEnvMapBlockBuffer.get(), blockOffset * sizeof(float), pTemp.get(), 0, sizeof(float) * sz);
            }
            // copy it back to the buffer
            prevEnvMapNumBlockX = envMapNumBlockX;
            prevEnvMapNumBlockY = envMapNumBlockY;
        }
    }

    // pRenderContext->clearUAV(mpCounters->getUAV().get(), uint4(0));
    // if(mCalculateCounters)
    {
        pRenderContext->clearUAV(mpPriorCounters->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(mpInitialCounters->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(mpTemporalCounters->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(mpSpatialCounters->getUAV().get(), uint4(0));
    }
    traceReceiver(pRenderContext);
    for(int i = 0; i < numPasses; i++)
    {
        if(mOptions.usePriorDistribution && useOurs)
        {
            buildPrior(pRenderContext, pSMS, emissiveSampler, envMapSampler, i);
        }
        initialSampling(pRenderContext, pVbuffer, pSMS, emissiveSampler, envMapSampler, i);
        if (mOptions.useTemporalResampling && mFrameIndex != 0)
        {
            temporalResampling(pRenderContext, pMotionVectors, pSMS, i);
        }
        if (mOptions.useSpatialResampling)
        {
            spatialResampling(pRenderContext, pSMS, i);
        }
    }
    resolve(pRenderContext);
}

void PSMSReSTIRPass::initialSampling(RenderContext* pRenderContext, const ref<Texture>& pVbuffer, const std::unique_ptr<SMS>& pSMS, const std::unique_ptr<EmissiveLightSampler>& emissiveSampler, const std::unique_ptr<EnvMapSampler>& envMapSampler, int passId)
{
    FALCOR_PROFILE(pRenderContext, "Initial Sampling");
    mpInitialSamplingPass->addDefine("USE_OURS", useOurs ? "1" : "0");
    mpInitialSamplingPass->addDefine("PRIOR_THREAD_BLOCK_SIZE", std::to_string(mOptions.buildPriorThreadGroupSize));
    mpInitialSamplingPass->addDefine("USE_CONSTRAINT", mOptions.useConstraint ? "1" : "0");
    mpInitialSamplingPass->addDefine("USE_TILING", mOptions.useTiling ? "1" : "0");
    mpInitialSamplingPass->addDefine("USE_PRIOR", mOptions.usePriorDistribution ? "1" : "0");
    mpInitialSamplingPass->addDefine("NUM_TILES_X", std::to_string(mOptions.numTilesX));
    mpInitialSamplingPass->addDefine("UNIFORM_THRESHOLD", std::to_string(mOptions.uniformThreshold));
    mpInitialSamplingPass->addDefine("PRIOR_THRESHOLD", std::to_string(mOptions.priorThreshold));
    mpInitialSamplingPass->addDefine("ALPHA", std::to_string(mOptions.alpha));
    mpInitialSamplingPass->addDefine("USE_BOUND_PROB", mOptions.useBoundProb ? "1" : "0");
    mpInitialSamplingPass->addDefine("USE_DIRECTIONAL", std::to_string(mOptions.useDirectional));
    // Bind resources.
    auto rootVar = mpInitialSamplingPass->getRootVar();
    mpScene->bindShaderData(rootVar["gScene"]);
    pSMS->bindShaderData(rootVar["gSMS"], mOptions.numTilesX);
    mpPixelDebug->prepareProgram(mpInitialSamplingPass->getProgram(), rootVar);
    auto var = rootVar["CB"]["gInitialSampling"];
    var["params"].setBlob(mParams);
    
    var["vbuffer"] = pVbuffer;
    var["outputReservoirs"] = mpOutputReservoirs[passId];

    var["frameIndex"] = mFrameIndex;
    var["frameDim"] = mFrameDim;
    var["imageBlockDim"] = mOptions.imageBlockDim;
    var["debugOutput"] = mpDebugOutputTexture;

    // prior distribution
    var["receiverInfos"] = mpCurrentReceiverInfo;

    var["solutionTiles"] = mpSolutionTiles[passId];
    var["passId"] = passId;

    var["calculateCounters"] = mCalculateCounters;
    var["initialCounters"] = mpInitialCounters;

    var["importanceMapDim"] = importanceMapDim;
    var["envMapBlockBuffer"] = mpEnvMapBlockBuffer;
    var["envMapNumBlockX"] = envMapNumBlockX;
    var["envMapNumBlockY"] = envMapNumBlockY;

    if (envMapSampler) envMapSampler->bindShaderData(var["envMapSampler"]);
    if (emissiveSampler) emissiveSampler->bindShaderData(var["emissiveSampler"]);
    // Dispatch.
    mpInitialSamplingPass->execute(pRenderContext, { mParams.screenTiles.x * kScreenTileDim.x, mParams.screenTiles.y * kScreenTileDim.y, 1u });
}

void PSMSReSTIRPass::temporalResampling(RenderContext* pRenderContext, const ref<Texture>& pMotionVectors, const std::unique_ptr<SMS>& pSMS, int passId)
{
    FALCOR_PROFILE(pRenderContext, "Temporal Resampling");

    mpTemporalResamplingPass->addDefine("USE_DIRECTIONAL", std::to_string(mOptions.useDirectional));
    // Bind resources.
    auto rootVar = mpTemporalResamplingPass->getRootVar();
    mpScene->bindShaderData(rootVar["gScene"]);
    pSMS->bindShaderData(rootVar["gSMS"]);

    auto var = rootVar["CB"]["gTemporalResampling"];
    var["params"].setBlob(mParams);
    var["motionVectors"] = pMotionVectors;
    var["temporalReservoirs"] = mpTemporalReservoirs[passId];
    var["outputReservoirs"] = mpOutputReservoirs[passId];
    var["temporalHistoryLength"] = 20.f;

    var["debugOutput"] = mpDebugOutputTexture;
    var["passId"] = passId;
    var["frameIndex"] = mFrameIndex;
    var["frameDim"] = mFrameDim;

    var["calculateCounters"] = mCalculateCounters;
    var["temporalCounters"] = mpTemporalCounters;
    // Dispatch.
    mpTemporalResamplingPass->execute(pRenderContext, { mParams.screenTiles.x * kScreenTileDim.x, mParams.screenTiles.y * kScreenTileDim.y, 1u });
}

void PSMSReSTIRPass::spatialResampling(RenderContext* pRenderContext, const std::unique_ptr<SMS>& pSMS, int passId)
{
    FALCOR_PROFILE(pRenderContext, "Spatial Resampling");

    mpSpatialResamplingPass->addDefine("USE_DIRECTIONAL", std::to_string(mOptions.useDirectional));

    // Bind resources.
    auto rootVar = mpSpatialResamplingPass->getRootVar();
    mpScene->bindShaderData(rootVar["gScene"]);
    pSMS->bindShaderData(rootVar["gSMS"]);

    auto var = rootVar["CB"]["gSpatialResampling"];
    std::swap(mpTemporalReservoirs[passId], mpOutputReservoirs[passId]);

    // clear output reservoirs
    pRenderContext->clearUAV(mpOutputReservoirs[passId]->getUAV().get(), uint4(0));
    var["params"].setBlob(mParams);
    var["neighborOffsets"] = mpNeighborOffsets;
    var["receiverInfos"] = mpCurrentReceiverInfo;
    var["inputReservoirs"] = mpTemporalReservoirs[passId];
    var["outputReservoirs"] = mpOutputReservoirs[passId];
    var["neighborCount"] = mOptions.mSpatialNeighborCount;
    var["gatherRadius"] = mOptions.mSpatialGatherRadius;

    var["debugOutput"] = mpDebugOutputTexture;

    var["reuseMaxIterations"] = mOptions.reuseMaxIterations;

    var["passId"] = passId;
    var["frameIndex"] = mFrameIndex;
    var["frameDim"] = mFrameDim;

    var["calculateCounters"] = mCalculateCounters;
    var["spatialCounters"] = mpSpatialCounters;
    // Dispatch.
    mpSpatialResamplingPass->execute(
        pRenderContext, { mParams.screenTiles.x * kScreenTileDim.x, mParams.screenTiles.y * kScreenTileDim.y, 1u });
}

void PSMSReSTIRPass::resolve(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "Resolve");

    // Bind resources.
    auto rootVar = mpResolvePass->getRootVar();
    mpPixelDebug->prepareProgram(mpResolvePass->getProgram(), rootVar);
    auto var = rootVar["gResolve"];
    var["debugOutput"] = mpDebugOutputTexture;
    for(int i = 0; i < numPasses; i++)
    {
        var["reservoirs"][i] = mpOutputReservoirs[i]; 
    }
    var["finalThp"] = mpFinalThp;

    var["numPasses"] = numPasses;
    var["frameIndex"] = mFrameIndex;
    var["frameDim"] = mFrameDim;

    var["calculateCounters"] = mCalculateCounters;

    var["priorCounters"] = mpPriorCounters;
    var["initialCounters"] = mpInitialCounters;
    var["temporalCounters"] = mpTemporalCounters;
    var["spatialCounters"] = mpSpatialCounters;

    mpResolvePass->execute(pRenderContext, {mFrameDim.x, mFrameDim.y, 1u});
}

void PSMSReSTIRPass::endFrame(RenderContext* pRenderContext)
{
    ++mFrameIndex;
    std::swap(mpTemporalReservoirs, mpOutputReservoirs);
    mpPixelDebug->endFrame(pRenderContext);
}

bool PSMSReSTIRPass::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.var("MAX_BERNOULLI_TRIALS", mOptions.maxBernoulliTrials, 1, 1024);

    if (widget.button("Clean Reservoirs"))
    {
        mFrameIndex = 0;
    }

    if (auto group = widget.group("Debugging"))
    {
        mpPixelDebug->renderUI(group);
    }

    widget.var("Num Passes", numPasses, 1, maxNumPasses);
    widget.checkbox("Calculate Counters", mCalculateCounters);
    widget.checkbox("Use Directional Light", mOptions.useDirectional);
    
    // initial resampling
    if(auto group = widget.group("Initial Sampling"))
    {
        dirty |= widget.checkbox("Use Ours", useOurs);
        if(useOurs)
        {
            if(auto subGroup = widget.group("Our Method Options"))
            {
                dirty |= widget.checkbox("Use Tiling", mOptions.useTiling);
                dirty |= widget.checkbox("Use Prior Distribution", mOptions.usePriorDistribution);
                dirty |= widget.var("Number of Tiles", mOptions.numTilesX, 1, 64);
                dirty |= widget.var("Uniform Threshold", mOptions.uniformThreshold, 1, 4);
                dirty |= widget.var("Prior Threshold", mOptions.priorThreshold, 1, 2);
                dirty |= widget.checkbox("Use Constraint", mOptions.useConstraint);
                dirty |= widget.checkbox("Use Bound Prob", mOptions.useBoundProb);
                dirty |= widget.var("Alpha", mOptions.alpha, 0.f, 1.f);
                mRecompile |= group.var("Image Block Size", mOptions.imageBlockDim);
                mRecompile |= group.var("Build Prior Thread Group Size", mOptions.buildPriorThreadGroupSize, 64, 256);
                dirty |= widget.var("Num Threads Used For Prior", mOptions.numThreadsUsedForPrior, 1, 256);

                if(mOptions.useDirectional)
                {
                    if(auto envGroup = widget.group("Environment Map Importance Sampling"))
                    {
                        dirty |= widget.var("Num Light Blocks X", envMapNumBlockX, 1, 32);
                        dirty |= widget.var("Num Light Blocks Y", envMapNumBlockY, 1, 32);
                    }
                }
            }
        }
    }
    if(auto group = widget.group("Temporal Resampling"))
    {
        dirty |= widget.checkbox("Temporal Resampling", mOptions.useTemporalResampling);
    }
    if(auto group = widget.group("Spatial Resampling"))
    {
        dirty |= widget.checkbox("Spatial Resampling", mOptions.useSpatialResampling);
        dirty |= widget.var("Spatial Neighbor Count", mOptions.mSpatialNeighborCount, 1u, 8u);
        dirty |= widget.var("Spatial Gather Radius", mOptions.mSpatialGatherRadius, 1.f, 30.f);
        dirty |= widget.var("Reuse Max Iterations", mOptions.reuseMaxIterations, -1, 20);
    }
    return dirty;
}

void PSMSReSTIRPass::setReSTIRParams(int useFixedSeed, uint fixedSeed,
    float lodBias, float specularRoughnessThreshold, uint2 frameDim, uint2 screenTiles, uint frameCount, uint seed)
{
    mParams.useFixedSeed = useFixedSeed;
    mParams.fixedSeed = fixedSeed;
    mParams.lodBias = lodBias;
    mParams.specularRoughnessThreshold = specularRoughnessThreshold;
    mParams.frameDim = frameDim;
    mParams.screenTiles = screenTiles;
    mParams.frameCount = frameCount;
    mParams.seed = seed;
}

ref<Texture> PSMSReSTIRPass::createNeighborOffsetTexture(uint32_t sampleCount)
{
    std::unique_ptr<int8_t[]> offsets(new int8_t[sampleCount * 2]);
    const int R = 254;
    const float phi2 = 1.f / 1.3247179572447f;
    float u = 0.5f;
    float v = 0.5f;
    for (uint32_t index = 0; index < sampleCount * 2;)
    {
        u += phi2;
        v += phi2 * phi2;
        if (u >= 1.f)
            u -= 1.f;
        if (v >= 1.f)
            v -= 1.f;

        float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
        if (rSq > 0.25f)
            continue;

        offsets[index++] = int8_t((u - 0.5f) * R);
        offsets[index++] = int8_t((v - 0.5f) * R);
    }

    return mpDevice->createTexture1D(sampleCount, ResourceFormat::RG8Snorm, 1, 1, offsets.get());
}
