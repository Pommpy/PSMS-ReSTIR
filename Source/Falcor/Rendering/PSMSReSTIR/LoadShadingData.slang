#include "Scene/SceneDefines.slangh"
__exported import Scene.Shading;
import Scene.Material.ShadingUtils;
import Rendering.Materials.TexLODHelpers;
import Utils.Math.MathHelpers;
import Rendering.PSMSReSTIR.Params;

/** Setup vertex data based on hit point.
    \param[in] hit Hit information.
    \param[out] materialID Material ID.
    \param[out] sdg Shading differential geometry.
    \return VertexData struct.
*/
VertexData loadVertexData(const HitInfo hit, out uint materialID, out ShadingDifferentialGeometry sdg, bool isPrevFrame = false)
{
    VertexData v = {};
    materialID = {};
    sdg = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        StaticVertexData vertices[3];
        v = gScene.getVertexData(triangleHit, vertices, isPrevFrame);
        sdg = gScene.getDifferentialGeometry(vertices, triangleHit.getBarycentricWeights());
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
    return v;
}

/** Setup vertex data based on hit point.
    \param[in] hit Hit information.
    \param[out] materialID Material ID.
    \param[out] sdg Shading differential geometry.
    \return VertexData struct.
*/
VertexData loadVertexData(const HitInfo hit, out uint materialID, bool isPrevFrame = false)
{
    VertexData v = {};
    materialID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        StaticVertexData vertices[3];
        v = gScene.getVertexData(triangleHit, vertices, isPrevFrame);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
    return v;
}

/** Setup vertex position based on hit point.
    \param[in] hit Hit information.
    \return float3 position in world space.
*/
float3 loadVertexPosition(const HitInfo hit, bool isPrevFrame = false)
{
    VertexData v = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        StaticVertexData vertices[3];
        v = gScene.getVertexData(triangleHit, vertices, isPrevFrame);
    }
#endif
    return v.posW;
}

/** Setup ShadingData based on loaded vertex/material attributes for a hit point.
    \param[in] hit Hit information.
    \param[in] rayDir Normalized ray direction.
    \return ShadingData struct.
*/
ShadingData loadShadingData(const HitInfo hit, const float3 rayDir, bool isPrevFrame = false)
{
    uint materialID = {};

    VertexData v = loadVertexData(hit, materialID);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);
    return sd;
}

/** Setup ShadingData based on loaded vertex/material attributes for a hit point.
    \param[in] hit Hit information.
    \param[in] rayDir Normalized ray direction.
    \param[out] sdg Shading differential geometry.
    \return ShadingData struct.
*/
ShadingData loadShadingData(const HitInfo hit, const float3 rayDir, out ShadingDifferentialGeometry sdg, bool isPrevFrame = false)
{
    uint materialID = {};

    VertexData v = loadVertexData(hit, materialID, sdg, isPrevFrame);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);
    return sd;
}

/** Determine hints to use when creating the material instance.
 */
uint getMaterialInstanceHints(const HitInfo hit, const bool isPrimary)
{
    uint hints = 0;
    if (hit.getType() == HitType::Triangle || hit.getType() == HitType::DisplacedTriangle)
    {
#if GBUFFER_ADJUST_SHADING_NORMALS
        if (kAdjustShadingNormals || isPrimary) hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;
#else
        if (kAdjustShadingNormals && !isPrimary) hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;
#endif
    }
    return hints;
}
