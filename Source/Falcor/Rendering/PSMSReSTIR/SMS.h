#pragma once
#include "Scene/Scene.h"
#include <Core/API/RenderContext.h>
#include <Core/Program/ShaderVar.h>
#include "Utils/Debug/PixelDebug.h"

namespace Falcor
{
    class FALCOR_API SMS
    {
    public:
        SMS(const ref<Scene>& pScene);

        void bindShaderData(const ShaderVar& var, const int numTilesX = 4) const;

        bool renderUI(Gui::Widgets& widget);

        void setupSpecularShapes(const ref<Scene>& pScene);
        void prepareResources();
        ref<Buffer> mpMaterialIDBuffer;
        ref<Buffer> mpSpecularAABBBuffer;
        ref<Buffer> mpIsUVSpaceSamplingBuffer;
    private:
        ref<Scene> mpScene;
        ref<Device> mpDevice;

        std::vector<uint32_t> materialIDs;
        std::vector<AABB> specularAABBs;
        std::vector<uint32_t> isUVSpaceSampling;
    };
}
