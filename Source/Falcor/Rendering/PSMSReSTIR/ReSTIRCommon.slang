import Params;
import Scene.HitInfo;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Utils.Math.MathHelpers;


bool isValidPackedHitInfo(PackedHitInfo packed)
{
    return packed.x != 0;
}


bool isPrimaryHitValid(Texture2D<PackedHitInfo> vbuffer, int2 pixel, int offset)
{
    HitInfo hit = HitInfo(vbuffer[pixel]);
    return hit.isValid();
}


bool isValidScreenRegion(ReSTIRPathTracerParams params, int2 pixel) { return all(pixel >= 0 && pixel < params.frameDim); }

// out_J = |inverse(T_i'(y))| -> jacobian from target to source domain
float pairwiseMIS_nonDefensiveNonCanonical(float cSum, float c0, float pi, float c1, float pc, float out_J = 1.0f)
{
    float pHatFrom_i = pi * out_J;
    return pc == 0.0f ? 0.0f : c0 * pHatFrom_i / ((cSum - c1) * pHatFrom_i + c1 * pc);
}

float pairwiseMIS_nonDefensiveCanonical(float cSum, float c0, float pi, float c1, float pc, float out_J = 1.0f)
{
    float w = c1 * pc;
    float pHatFrom_i = pi * out_J;
    return pc == 0.0f ? 0.0f : (c0 / (cSum - c1)) * w / ((cSum - c1) * pHatFrom_i + w);
    // return pc == 0.0f ? 0.0f : (c0 / cSum) * w / ((cSum - c1) * pi + w);
}

// Note: when using defensive version, canonical MIS weight should start with c1 / cSum
float pairwiseMIS_defensiveNonCanonical(float cSum, float c0, float pi, float c1, float pc, float out_J = 1.0f)
{
    float pHatFrom_i = pi * out_J;
    float w = (cSum - c1) * pHatFrom_i;
    return pc == 0.0f ? 0.0f : (c0 / cSum) * w / (w + c1 * pc);
}

float pairwiseMIS_defensiveCanonical(float cSum, float c0, float pi, float c1, float pc, float out_J = 1.0f)
{
    float pHatFrom_i = pi * out_J;
    float w = c1 * pc;
    return pc == 0.0f ? 0.0f : (c0 / cSum) * w / ((cSum - c1) * pHatFrom_i + w);
}
