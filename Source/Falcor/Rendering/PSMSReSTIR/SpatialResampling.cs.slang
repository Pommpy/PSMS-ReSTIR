import Rendering.PSMSReSTIR.Params;
import Scene.HitInfo;
import Reservoir;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import ReSTIRCommon;
import Utils.Sampling.SampleGenerator;
import LoadShadingData;
import Shift;
import Utils.Color.ColorHelpers;

struct SpatialResampling
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    ReSTIRPathTracerParams params;

    StructuredBuffer<ReceiverInfo> receiverInfos;
    Texture1D<float2> neighborOffsets;

    StructuredBuffer<Reservoir> inputReservoirs;
    RWStructuredBuffer<Reservoir> outputReservoirs;

    bool calculateCounters;
    RWStructuredBuffer<uint> spatialCounters;

    RWTexture2D<float4> debugOutput; ///< Debug output texture.

    uint neighborCount;
    float gatherRadius;
    int reuseMaxIterations;

    int frameIndex;
    int passId;
    int2 frameDim;

    struct NeighborValidMask
    {
        uint data;

        __init()
        {
            this.data = 0;
        }

        bool isValid(int i)
        {
            return data >> i & 1;
        }

        [mutating]
        void setValid(int i)
        {
            data |= 1 << i;
        }
    }

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, uint i)
    {
        int2 neighborPixel = int2(0, 0);

        uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
        neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * gatherRadius);

        return neighborPixel;
    }

    void resampling(uint2 pixel)
    {
        printSetPixel(pixel);
        logSetPixel(pixel);

        bool insideScreen = isValidScreenRegion(params, pixel);
        if (!insideScreen) return;
        int pixelIndex = pixel.y * frameDim.x + pixel.x;
        ReceiverInfo receiverInfo = receiverInfos[pixelIndex];
        if(!receiverInfo.valid) return;

        var sg = SampleGenerator(pixel, (1 + 1 + 1) * params.seed + (1 + 1 + 0) + frameIndex * 10 + passId);
        const uint centralOffset = params.getReservoirOffset(pixel);
        Reservoir dstReservoir = inputReservoirs[centralOffset];
        if (dstReservoir.M == 0.f)
        {
            outputReservoirs[centralOffset] = dstReservoir;
            return;
        }
        Reservoir centralReservoir = dstReservoir;
        float centralM = dstReservoir.M;
        dstReservoir.init();
        NeighborValidMask neighborValidMask = {};
        float cSum = centralM;
        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;
        for (uint i = 0; i < neighborCount; ++i)
        {
            int2 neighborPixel = getNextNeighborPixel(startIndex, pixel, i);
            if (!isValidScreenRegion(params, neighborPixel)) continue;
            Reservoir neighborReservoir = inputReservoirs[params.getReservoirOffset(neighborPixel)];
            if (neighborReservoir.M == 0.f) continue;
            cSum += neighborReservoir.M;
            neighborValidMask.setValid(i);
        }

        float centralMISWeight = centralM / cSum;
        for (uint i = 0; i < neighborCount; ++i)
        {
            if (!neighborValidMask.isValid(i)) continue;
            int2 neighborPixel = getNextNeighborPixel(startIndex, pixel, i);
            Reservoir neighborReservoir = inputReservoirs[params.getReservoirOffset(neighborPixel)];

            float3 Tp = float3(0.f);
            float3 Tp2 = float3(0.f);
            SMSInfo smsInfo = neighborReservoir.smsInfo;
            SMSInfo smsInfo2 = centralReservoir.smsInfo;

            uint3 shiftNewtons1 = uint3(0);
            uint3 shiftNewtons2 = uint3(0);
            Tp = shiftPathSMS(centralReservoir, smsInfo, sg, shiftNewtons1, false, false, reuseMaxIterations);
            Tp2 = shiftPathSMS(neighborReservoir, smsInfo2, sg, shiftNewtons2, false, false, reuseMaxIterations);
            if (calculateCounters)
            {
                InterlockedAdd(spatialCounters[0], shiftNewtons1.x);
                InterlockedAdd(spatialCounters[0], shiftNewtons2.x);
                InterlockedAdd(spatialCounters[1], shiftNewtons1.y);
                InterlockedAdd(spatialCounters[1], shiftNewtons2.y);
                InterlockedAdd(spatialCounters[2], shiftNewtons1.z);
                InterlockedAdd(spatialCounters[2], shiftNewtons2.z);
            }
            neighborReservoir.smsInfo = smsInfo2;
            float neighborMISWeight = pairwiseMIS_defensiveNonCanonical(cSum, neighborReservoir.M, luminance(neighborReservoir.F), centralM, luminance(Tp2));
            dstReservoir.merge(neighborReservoir, Tp2, neighborMISWeight, sg);

            centralMISWeight += pairwiseMIS_defensiveCanonical(cSum, neighborReservoir.M, luminance(Tp), centralM, luminance(centralReservoir.F));
        }
        dstReservoir.merge(centralReservoir, centralReservoir.F, centralMISWeight, sg);

        dstReservoir.M = cSum;
        dstReservoir.finalizeGRIS();
        dstReservoir.smsInfo.hit = centralReservoir.smsInfo.hit;
        dstReservoir.smsInfo.dir = centralReservoir.smsInfo.dir;
        dstReservoir.pathLength = centralReservoir.pathLength;
        outputReservoirs[centralOffset] = dstReservoir;
    }
}

cbuffer CB
{
    SpatialResampling gSpatialResampling;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gSpatialResampling.resampling(dispatchThreadId.xy);
}
