#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR


struct SpecularShapeData
{
    uint        instanceID          = kInvalidIndex;    ///< Geometry instance ID in the scene.
    uint        triangleOffset      = kInvalidIndex;    ///< Offset into LightCollection's global list of emissive triangles.
    uint        triangleCount       = 0;                ///< Number of triangles in mesh light.
    uint        materialID          = kInvalidIndex;    ///< Material ID (index into scene.materials array).

    static constexpr uint kInvalidIndex = 0xffffffff;
};


END_NAMESPACE_FALCOR
