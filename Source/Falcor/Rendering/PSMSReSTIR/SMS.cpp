#include "SMS.h"

namespace Falcor
{
    SMS::SMS(const ref<Scene>& pScene)
        : mpScene(pScene)
        , mpDevice(mpScene->getDevice())
    {
        FALCOR_ASSERT(mpScene);

        setupSpecularShapes(mpScene);
    }

    void SMS::setupSpecularShapes(const ref<Scene>& pScene)
    {
        for (uint32_t instanceID = 0; instanceID < pScene->getGeometryInstanceCount(); instanceID++)
        {
            const GeometryInstanceData& instanceData = pScene->getGeometryInstance(instanceID);

            // We only support triangle meshes.
            if (instanceData.getType() != GeometryType::TriangleMesh)
                continue;

            auto pMaterial = pScene->getMaterial(MaterialID::fromSlang(instanceData.materialID))->toBasicMaterial();

            AABB aabb = pScene->getMeshBounds(instanceData.geometryID);

            if (pMaterial && pMaterial->isCausticBouncer())
            {
                materialIDs.push_back(instanceData.materialID);
                specularAABBs.push_back(aabb);
                bool isUVSpace = pMaterial->isUVSpaceSampling();
                isUVSpaceSampling.push_back(isUVSpace ? 1 : 0);
            }
        }
    }
 
    void SMS::bindShaderData(const ShaderVar& var, const int numTilesX)const
    {
        if(materialIDs.empty())
            return;
        var["specularAABBs"] = mpSpecularAABBBuffer;
        var["specularMaterialIDs"] = mpMaterialIDBuffer;
        var["isUVSpaceSampling"] = mpIsUVSpaceSamplingBuffer;
        var["specularShapesCount"] = mpMaterialIDBuffer->getElementCount();
        var["numTilesX"] = numTilesX;
    }

    void SMS::prepareResources()
    {
        uint32_t elementCount(materialIDs.size());

        if (elementCount != 0 && (!mpMaterialIDBuffer || mpMaterialIDBuffer->getElementCount() < elementCount))
        {
            mpMaterialIDBuffer = mpDevice->createStructuredBuffer(
                sizeof(uint32_t),
                elementCount,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                materialIDs.data(),
                false
            );
        }
        if (elementCount != 0 && (!mpSpecularAABBBuffer || mpSpecularAABBBuffer->getElementCount() < elementCount))
        {
            mpSpecularAABBBuffer = mpDevice->createStructuredBuffer(
                sizeof(AABB),
                elementCount,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                specularAABBs.data(),
                false
            );
        }
        if(elementCount != 0 && (!mpIsUVSpaceSamplingBuffer || mpIsUVSpaceSamplingBuffer->getElementCount() < elementCount))
        {
            mpIsUVSpaceSamplingBuffer = mpDevice->createStructuredBuffer(
                sizeof(uint32_t),
                elementCount,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                isUVSpaceSampling.data(),
                false
            );
        }
    }

    bool SMS::renderUI(Gui::Widgets& widget)
    {
        return false;
    }
}
