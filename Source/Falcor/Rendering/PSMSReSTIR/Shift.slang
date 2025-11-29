import Reservoir;
import Scene.RaytracingInline;
import SMS;
import Rendering.PSMSReSTIR.LoadShadingData;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.EmissiveLightSamplerInterface;

#ifndef USE_DIRECTIONAL
#define USE_DIRECTIONAL 0
#endif

float3 shiftPathSMS<S : ISampleGenerator>(Reservoir srcReservoir, inout SMSInfo smsInfo, inout S sg, inout uint3 shiftNewtons, bool toPrevFrame = false, bool isTemporalResampling = false, int maxIterations = -1)
{
    shiftNewtons = uint3(0);
    uint newton_iterations = 0;
    if (srcReservoir.weight == 0.f) return float3(0.f);

    ShadingDifferentialGeometry sdg;
    const ShadingData sd = loadShadingData(smsInfo.hit, smsInfo.dir, sdg);

    // TODO: support `lodBias`
    let lod = ExplicitLodTextureSampler(0.f);
    const bool isPrimaryHit = srcReservoir.pathLength == 0;
    let hints = getMaterialInstanceHints(smsInfo.hit, isPrimaryHit);
    const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);

    float3 lightPosOrDir = srcReservoir.smsInfo.lightPosOrDir;
    const uint triangleIndex = srcReservoir.smsInfo.lightData & kLightIndexMask;
    const float2 uv = float2(srcReservoir.smsInfo.uvData & 0xffff, srcReservoir.smsInfo.uvData >> 16) / float(0xffff);
    TriangleLightSample tls;
    if (USE_DIRECTIONAL == 0)
    {
        // get light data
        sampleTriangle(sd.posW, triangleIndex, uv, tls, toPrevFrame);
        lightPosOrDir = tls.posW;
    }

    float3 specularPos = srcReservoir.smsInfo.firstSpecularPos;
    const float3 neighborPos = loadVertexPosition(srcReservoir.smsInfo.hit, isTemporalResampling & !toPrevFrame);

    ManifoldVertex si = ManifoldVertex(sd);
    si.dp_du = sdg.dpdu;
    si.dp_dv = sdg.dpdv;
    si.dn_du = sdg.dndu;
    si.dn_dv = sdg.dndv;
    float3 baseEndpoint = si.p;
    // shift the base path
    float3 originalDir = normalize(specularPos - neighborPos);
    ManifoldVertex offsetPath[maxBounces];
    uint seedMaterialID[maxBounces] = {};
    uint bounces;
    float3 initDir = float3(0.f);
    int2 success_counter = gSMS.samplePath<S, USE_DIRECTIONAL>( baseEndpoint, lightPosOrDir, sg, true, offsetPath, seedMaterialID, bounces, specularPos, initDir, -1, -1, -1, true, float2(0.f), float2(1.f), toPrevFrame, maxIterations );

    shiftNewtons.x += 1;
    shiftNewtons.z += success_counter.y;
    bool success = success_counter.x == 1;
    if (!success)
    {
        return float3(0.f);
    }
    float3 offsetSpecularPos = offsetPath[0].p;
    float3 offsetDir = normalize(offsetPath[0].p - baseEndpoint);

    float3 specularVal;
    switch (bounces)
    {
    case 1:
        specularVal = gSMS.evalSpecularContribution<1, USE_DIRECTIONAL>(si, lightPosOrDir, offsetPath, toPrevFrame);
        break;
    case 2:
        specularVal = gSMS.evalSpecularContribution<2, USE_DIRECTIONAL>(si, lightPosOrDir, offsetPath, toPrevFrame);
        break;
    default:
        specularVal = float3(0.f);
    }
    if (all(specularVal == 0.f))
    {
        return float3(0.f);
    }

    float3 bsdfVal = mi.eval(sd, offsetDir, sg);
    // check the invertibility
    if (isTemporalResampling && USE_DIRECTIONAL == 0)
    {
        sampleTriangle(neighborPos, triangleIndex, uv, tls, !toPrevFrame);
        lightPosOrDir = tls.posW;
    }
    success_counter = gSMS.samplePath<S, USE_DIRECTIONAL>( neighborPos, lightPosOrDir, sg, true, offsetPath, seedMaterialID, bounces, offsetSpecularPos, initDir, -1, -1, -1, true, float2(0.f), float2(1.f), isTemporalResampling & !toPrevFrame, maxIterations );
    shiftNewtons.z += success_counter.y;

    success = success_counter.x == 1;
    if (!success)
    {
        return float3(0.f);
    }
    float3 baseDir = normalize(offsetPath[0].p - neighborPos);
    if (abs(dot(originalDir, baseDir) - 1.f) > uniqueness_threshold)
    {
        return float3(0.f);
    }

    specularPos = offsetSpecularPos;

    // bijective mappings
    shiftNewtons.y += 1;
    float3 Thp = bsdfVal * specularVal;

    smsInfo.firstSpecularPos = specularPos;
    smsInfo.lightPosOrDir = srcReservoir.smsInfo.lightPosOrDir;
    smsInfo.lightData = srcReservoir.smsInfo.lightData;
    smsInfo.uvData = srcReservoir.smsInfo.uvData;
    smsInfo.Le = srcReservoir.smsInfo.Le;
    float3 Le = srcReservoir.smsInfo.Le;
    if (USE_DIRECTIONAL == 0)
    {
        Le = tls.Le;
    }
    return Thp * Le;
}

