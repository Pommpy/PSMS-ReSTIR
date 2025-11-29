#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
struct WriteToEnvMapBuffer
{
    RWStructuredBuffer<float> envMapImportanceBuffer;

    int envMapNumBlockX;
    int envMapNumBlockY;
    int2 importanceMapDim;

    // importance map texel
    void process(uint2 texel)
    {
        if (any(texel >= importanceMapDim)) return;

        int2 numBlocks = int2(envMapNumBlockX, envMapNumBlockY);
        int2 blockSize = (importanceMapDim + numBlocks - 1) / numBlocks; // ceil

        int2 block = int2(texel / uint2(blockSize));
        block = clamp(block, int2(0), numBlocks - 1);

        int blockStride = blockSize.x * blockSize.y;
        int blockId = block.y * envMapNumBlockX + block.x;

        int2 local = int2(texel) - block * blockSize;

        int texelIndex = local.y * blockSize.x + local.x;
        int linearIndex = blockId * blockStride + texelIndex;

        int2 dimension = gScene.envMap.getDimensions();
        // sample env map

        int numSamples = 8;
        float L = 0.f;
        int2 outputDimInSamples = importanceMapDim * numSamples;
        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < numSamples; j++)
            {
                float2 samplePos = texel * numSamples + int2(j, i) + 0.5;
                float2 p = 1.f * samplePos / outputDimInSamples;
                float3 dir = oct_to_ndir_equal_area_unorm(p);
                float2 uv = world_to_latlong_map(dir);

                float3 radiance = gScene.envMap.envMap.SampleLevel(gScene.envMap.envSampler, uv, 0).rgb;
                L += luminance(radiance);
            }
        }
        envMapImportanceBuffer[linearIndex] = L / (numSamples * numSamples);
    }
}

ConstantBuffer<WriteToEnvMapBuffer> gWriteToEnvMapBuffer;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gWriteToEnvMapBuffer.process(dispatchThreadId.xy);
}
