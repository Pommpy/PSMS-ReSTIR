import Rendering.PSMSReSTIR.Params;
import Reservoir;

struct ReSTIR
{
    ReSTIRPathTracerParams params;
    // StructuredBuffer<Reservoir> reservoirs;
    Texture2D<float4> finalThp;

    float3 getFinalSample(const uint2 pixel)
    {
        return finalThp[pixel].rgb;
    }
}

ParameterBlock<ReSTIR> gReSTIR;
