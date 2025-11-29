#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Scene.RaytracingInline;
import Utils.Debug.PixelDebug;
import Utils.Math.MathHelpers;
import LoadShadingData;
import Utils.Color.ColorHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;
import Reservoir;

struct TraceReceiver
{
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    RWTexture2D<float4> debugOutput;
    uint2 frameDim;
    uint frameIndex;
    RWStructuredBuffer<ReceiverInfo> receiverInfos;
    int maxDepth;

    uint getPixelIndex(uint2 pixel) { return pixel.y * frameDim.x + pixel.x; }

    void traceReceiverInfo(Ray ray, inout ReceiverInfo receiverInfo, inout TinyUniformSampleGenerator sg)
    {
        // keep tracing rays
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        for (int i = 0; i < maxDepth; i++)
        {
            float hitT;
            const HitInfo hit = sceneRayQuery.traceRay(ray, hitT);
            if (hit.isValid())
            {
                ShadingData sd = loadShadingData(hit, ray.dir);
                if (sd.mtl.isCausticReceiver())
                {
                    receiverInfo.valid = true;
                    receiverInfo.hit = hit;
                    receiverInfo.V = -ray.dir;
                    receiverInfo.length = i;
                    return;
                }
                // Create texture sampler.
                let lod = ExplicitLodTextureSampler(0.f);
                // Create BSDF instance and query its properties.
                let hints = getMaterialInstanceHints(hit, true /* primary hit */);
                let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
                let bsdfProperties = mi.getProperties(sd);
                // Check for BSDF lobesTypes that ReSTIR can sample.
                uint lobesTypes = mi.getLobeTypes(sd);
                const bool hasDeltaLobes = (lobesTypes & LobeType::Delta) != 0;
                if (sd.mtl.isCausticBouncer())
                {
                    if (sd.mtl.getIoR() == 1.f) sd.mtl.setActiveLobes((uint)LobeType::DeltaReflection);
                    else sd.mtl.setActiveLobes((uint)LobeType::DeltaTransmission);
                }
                BSDFSample result;
                if (!hasDeltaLobes || !mi.sample(sd, sg, result))
                {
                    return;
                }
                bool isReflection = result.isLobe(LobeType::Reflection);
                ray.origin = sd.posW;
                ray.dir = result.wo;
                ray.tMin = 0.001f;
                ray.tMax = 1e30f;
            }
        }
    }

    void process(uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;
        printSetPixel(pixel);
        print("Tracing Receivers ------------------------", 1);

        debugOutput[pixel] = float4(0.f);
        uint pixelIndex = getPixelIndex(pixel);

        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, frameIndex);
        Ray ray = gScene.camera.computeRayPinhole(pixel, frameDim, false);
        ReceiverInfo receiverInfo;
        receiverInfo.valid = false;
        traceReceiverInfo(ray, receiverInfo, sg);
        if(receiverInfo.valid)
        {
            debugOutput[pixel] = float4(0.f, 1.f, 0.f, 1.f);
        }
        receiverInfos[pixelIndex] = receiverInfo;
    }
}

ConstantBuffer<TraceReceiver> gTraceReceiver;

// trace secondary hits
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gTraceReceiver.process(dispatchThreadId.xy);
}
