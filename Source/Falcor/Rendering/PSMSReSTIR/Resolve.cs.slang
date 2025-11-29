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

#define MAX_RESTIR_PASSES 4

struct Resolve
{
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    RWTexture2D<float4> debugOutput;
    RWTexture2D<float4> finalThp;
    uint2 frameDim;
    uint frameIndex;

    int numPasses;

    bool calculateCounters;
    RWStructuredBuffer<uint> priorCounters;
    RWStructuredBuffer<uint> initialCounters;
    RWStructuredBuffer<uint> temporalCounters;
    RWStructuredBuffer<uint> spatialCounters;

    StructuredBuffer<Reservoir> reservoirs[MAX_RESTIR_PASSES];
    bool isValid(uint2 pixel)
    {
        return all(pixel >= 0 && pixel < frameDim);
    }
    int getPixelIndex(uint2 pixel)
    {
        return pixel.y * frameDim.x + pixel.x;
    }

    void process(int2 pixel)
    {
        // merge multiple restir passes
        if(!isValid(pixel)) return;
        float4 finalColor = float4(0.f);
        printSetPixel(pixel);
        print("Resolve -----------------------------------------------", 1);
        // print counters
        if (calculateCounters)
        {
            print("Prior Counters -----------------------------------------------------------", 1);
            print("Sample success rate: ", 1.0 * priorCounters[1] / priorCounters[0]);
            print("Newtons per sample: ", 1.0 * priorCounters[2] / max(priorCounters[0], 1));

            print("Initial Counters -----------------------------------------------------------", 1);
            print("Sample success rate: ", 1.0 * initialCounters[1] / initialCounters[0]);
            print("Newtons per sample: ", 1.0 * initialCounters[2] / max(initialCounters[0], 1));
            print("Bernoulli per solution: ", 1.0 * initialCounters[3] / max(initialCounters[1], 1));
            print("Newtons per solution: ", 1.0 * initialCounters[4] / max(initialCounters[1], 1));

            print("Temporal Counters -----------------------------------------------------------", 1);
            print("Bijective shift rate: ", 1.0 * temporalCounters[1] / temporalCounters[0]);
            print("Newtons per shift: ", 1.0 * temporalCounters[2] / max(temporalCounters[0], 1));

            print("Spatial Counters -----------------------------------------------------------", 1);
            print("Bijective shift rate: ", 1.0 * spatialCounters[1] / spatialCounters[0]);
            print("Newtons per shift: ", 1.0 * spatialCounters[2] / max(spatialCounters[0], 1));
        }

        int pixelIndex = getPixelIndex(pixel);
        for (int i = 0; i < numPasses; ++i)
        {
            Reservoir reservoir = reservoirs[i][pixelIndex];
            if (reservoir.M == 0.f) continue;

            // merge
            float3 sample = reservoir.F * reservoir.weight;
            finalColor += float4(sample, 1.f);
        }
        finalColor /= numPasses;

        // finalColor = float4(1.f);
        finalThp[pixel] = finalColor;
        // debugOutput[pixel] = finalColor;
    }
}

ConstantBuffer<Resolve> gResolve;

// trace secondary hits
[numthreads(16, 16)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gResolve.process(dispatchThreadId.xy);
}
