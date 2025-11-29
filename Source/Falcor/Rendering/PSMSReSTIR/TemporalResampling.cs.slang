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

struct TemporalResampling
{
    ReSTIRPathTracerParams params;

    Texture2D<float2> motionVectors;

    StructuredBuffer<Reservoir> temporalReservoirs;
    RWStructuredBuffer<Reservoir> outputReservoirs;

    RWTexture2D<float4> debugOutput; ///< Debug output texture.

    bool calculateCounters;
    RWStructuredBuffer<uint> temporalCounters;

    float temporalHistoryLength;

    int passId;
    int2 frameDim;
    int frameIndex;

    void resampling(uint2 pixel)
    {
        printSetPixel(pixel);
        logSetPixel(pixel);

        bool insideScreen = isValidScreenRegion(params, pixel);
        if (!insideScreen) return;

        // debugOutput[pixel] = float4(0.f, 0.f, 0.f, 1.f);

        var sg = SampleGenerator(pixel, (1 + 1 + 1) * params.seed + (1 + 1 + 0) + frameIndex * 10 + passId);

        const uint centralOffset = params.getReservoirOffset(pixel);
        Reservoir dstReservoir = outputReservoirs[centralOffset];
        if (dstReservoir.M == 0.f) return;
        Reservoir centralReservoir = dstReservoir;
        float centralM = dstReservoir.M;

        dstReservoir.init();

        // fetch temporal reservoir
        float2 motionVector = motionVectors[pixel];
        int2 prevPixel = pixel + motionVector * params.frameDim + sampleNext2D(sg);
        if (!isValidScreenRegion(params, prevPixel)) return;

        Reservoir temporalReservoir = temporalReservoirs[params.getReservoirOffset(prevPixel)];
        if (temporalReservoir.M == 0.f) return;
        float temporalM = min(temporalReservoir.M, temporalHistoryLength);

        // shift paths

        SMSInfo smsInfo = temporalReservoir.smsInfo;
        SMSInfo smsInfo2 = centralReservoir.smsInfo;
        float4 debugInfo1 = float4(0.f);
        float4 debugInfo2 = float4(0.f);

        uint3 shiftNewtons1 = uint3(0);
        uint3 shiftNewtons2 = uint3(0);
        float3 Tp = shiftPathSMS(centralReservoir, smsInfo, sg, shiftNewtons1, true, true);
        float3 Tp2 = shiftPathSMS(temporalReservoir, smsInfo2, sg, shiftNewtons2, false, true);

        if (calculateCounters)
        {
            InterlockedAdd(temporalCounters[0], shiftNewtons1.x);
            InterlockedAdd(temporalCounters[0], shiftNewtons2.x);

            InterlockedAdd(temporalCounters[1], shiftNewtons1.y);
            InterlockedAdd(temporalCounters[1], shiftNewtons2.y);

            InterlockedAdd(temporalCounters[2], shiftNewtons1.z);
            InterlockedAdd(temporalCounters[2], shiftNewtons2.z);
        }

        // compute MIS weight and merge
        float temporal_pHat = luminance(temporalReservoir.F);
        if (temporal_pHat > 0.f)
        {
            temporalReservoir.smsInfo = smsInfo2;
            float temporalMISWeight = (temporalM * temporal_pHat) / (temporalM * temporal_pHat + centralM * luminance(Tp2));
            bool selectTemporal = dstReservoir.merge(temporalReservoir, Tp2, temporalMISWeight, sg);
            if (selectTemporal && temporalM == 20)
            {
                debugOutput[pixel] = float4(0.f, 0.f, 1.f, 1.f);
            }
        }

        float central_pHat = luminance(centralReservoir.F);
        if (central_pHat > 0.f)
        {
            float centralMISWeight = (centralM * central_pHat) / (centralM * central_pHat + temporalM * luminance(Tp));
            bool selectCurrent = dstReservoir.merge(centralReservoir, centralReservoir.F, centralMISWeight, sg);
            if (selectCurrent)
            {
                // debugOutput[pixel] = float4(0.f, 1.f, 0.f, 1.f);
            }
        }
        dstReservoir.M = centralM + temporalM;
        dstReservoir.finalizeGRIS();
        // Note: Even if all reservoirs contain no admissible paths,
        // we still need to store the specular chain endpoint from the central reservoir.
        dstReservoir.smsInfo.hit = centralReservoir.smsInfo.hit;
        dstReservoir.smsInfo.dir = centralReservoir.smsInfo.dir;
        dstReservoir.pathLength = centralReservoir.pathLength;
        
        outputReservoirs[centralOffset] = dstReservoir;
    }
}

cbuffer CB
{
    TemporalResampling gTemporalResampling;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gTemporalResampling.resampling(dispatchThreadId.xy);
}
