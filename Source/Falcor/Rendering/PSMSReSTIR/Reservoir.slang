import LoadShadingData;
import Utils.Color.ColorHelpers;

static const uint kLightIndexBits = 30;
static const uint kLightIndexMask = (1 << kLightIndexBits) - 1;

struct SMSInfo
{
    HitInfo hit;
    float3 dir;
    float3 firstSpecularPos;
    float3 lightPosOrDir; // TODO: support analytic lights
    float3 Le;
    uint lightData; // TODO: support analytic lights
    uint uvData;

    __init()
    {
        hit = HitInfo();
        dir = float3(0.f);
        firstSpecularPos = float3(0.f);
        lightPosOrDir = float3(0.f);
        Le = float3(0.f);
        lightData = 0;
        uvData = 0;
    }
}

struct Reservoir
{
    float3 F;
    float weight;
    float M;
    uint pathLength;

    SMSInfo smsInfo;

    __init()
    {
        F = float3(0.f);
        weight = 0.f;
        M = 0.f;
        pathLength = 0;
        smsInfo = SMSInfo();
    }

    [mutating]
    void init()
    {
        F = float3(0.f);
        weight = 0.f;
        M = 0.f;
        pathLength = 0;
        smsInfo = SMSInfo();
    }

    [mutating]
    void add(float3 integrand, float invPdf, float3 firstSpecularPos, float3 lightPosOrDir, float3 Le, uint lightIndex, float2 uv)
    {
        this.F = integrand;
        this.weight = invPdf;
        this.smsInfo.firstSpecularPos = firstSpecularPos;
        this.smsInfo.lightPosOrDir = lightPosOrDir;
        this.smsInfo.Le = Le;
        this.smsInfo.lightData = lightIndex & kLightIndexMask;
        this.smsInfo.uvData = uint(saturate(uv.x) * 0xffff) | (uint(saturate(uv.y) * 0xffff) << 16);
    }

    [mutating]
    bool merge<S : ISampleGenerator>(Reservoir inReservoir, float3 F, float misWeight, inout S sg)
    {
        float w = misWeight * luminance(F) * inReservoir.weight;
        this.weight += w;
        bool selected = sampleNext1D(sg) * this.weight < w;
        if (selected)
        {
            this.F = F;
            this.smsInfo = inReservoir.smsInfo;
        }
        return selected;
    }

    [mutating]
    void finalizeGRIS()
    {
        float p_hat = luminance(F);
        this.weight = (p_hat == 0.f) ? 0.f : (this.weight / p_hat);
    }
}

struct ReceiverInfo
{
    bool valid; // this may not be needed
    int length;
    HitInfo hit;
    float3 V;
}
