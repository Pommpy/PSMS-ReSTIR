from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("PathTracer")
    gPathTracerParams = {
        'samplesPerPixel': 1,
        'usePSMSReSTIR': True,
        'PSMSReSTIROptions': {
            'maxBernoulliTrials': 128,
            'useTemporalResampling': True,
            'useSpatialResampling': True,
            'useTiling': True,
            'usePriorDistribution': True,
            'buildPriorThreadGroupSize': 128,
            'numThreadsUsedForPrior': 128,
            'numTilesX': 16,
            'uniformThreshold': 1,
            'priorThreshold': 1,
            'useConstraint': True,
            'useBoundProb': True,
            'solverThreshold': 1e-3,
            'useDirectional': False
        }
    }
    PathTracer = createPass("PathTracer", gPathTracerParams)
    g.addPass(PathTracer, "PathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': False, 'precisionMode': 'Double'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0, 'operator': 'Linear'})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    g.addEdge("VBufferRT.frameS", "PathTracer.frameS")
    g.addEdge("VBufferRT.frameT", "PathTracer.frameT")
    g.addEdge("VBufferRT.posU", "PathTracer.posU")
    g.addEdge("VBufferRT.posV", "PathTracer.posV")
    g.addEdge("VBufferRT.normU", "PathTracer.normU")
    g.addEdge("VBufferRT.normV", "PathTracer.normV")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    g.markOutput("AccumulatePass.output")
    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None

scene = "../scenes/ReflectivePlane/plane.pyscene"
m.loadScene(scene, buildFlags=(SceneBuilderFlags.DontMergeMaterials | SceneBuilderFlags.DontOptimizeMaterials))
m.scene.camera.animated = False
m.scene.animated = True