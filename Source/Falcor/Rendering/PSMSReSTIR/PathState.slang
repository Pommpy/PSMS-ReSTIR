__exported import Utils.Sampling.SampleGenerator;
__exported import Utils.Math.Ray;
import Scene.HitInfo;
import Reservoir;

static const float kRayTMax = 1e30f;

static const uint kPathLengthBitCount = 10u;
static const uint kPathLengthBitMask = (1u << kPathLengthBitCount) - 1u;
static const uint kPathFlagsBitCount = 32u - kPathLengthBitCount;
static const uint kPathFlagsBitMask = ((1u << kPathFlagsBitCount) - 1u) << kPathLengthBitCount;

/** Path flags. The path flags are currently stored in kPathFlagsBitCount bits.
*/
enum class PathFlags
{
    active                      = 0x0001,   ///< Path is active/terminated.
    hit                         = 0x0002,   ///< Result of the scatter ray (0 = miss, 1 = hit).
};

struct PathState
{
    float3 origin; // The origin of the ray
    float3 dir;    // The direction of the ray
    SampleGenerator sg; // The sample generator for the path
    HitInfo hit;        // The hit information for the path
    uint flagsAndPathLength;

    Reservoir reservoir;

    [mutating]
    void setHit(HitInfo hit)
    {
        this.hit = hit;
        flagsAndPathLength |= (uint(PathFlags.hit) << kPathLengthBitCount);
    }

    [mutating]
    void clearHit()
    {
        flagsAndPathLength &= ~(uint(PathFlags.hit) << kPathLengthBitCount);
    }

    bool isHit()
    {
        const uint bit = uint(PathFlags.hit) << kPathLengthBitCount;
        return (flagsAndPathLength & bit) != 0;
    }

    [mutating]
    void setActive()
    {
        flagsAndPathLength |= (uint(PathFlags.active) << kPathLengthBitCount);
    }

    [mutating]
    void terminate()
    {
        flagsAndPathLength &= ~(uint(PathFlags.active) << kPathLengthBitCount);
    }

    bool isActive()
    {
        const uint bit = uint(PathFlags.active) << kPathLengthBitCount;
        return (flagsAndPathLength & bit) != 0;
    }

    [mutating]
    void setPathLength(uint pathLength)
    {
        flagsAndPathLength = (flagsAndPathLength & kPathFlagsBitMask) | (pathLength & kPathLengthBitMask);
    }

    [mutating]
    void incrementPathLength()
    {
        ++flagsAndPathLength;
    }

    uint getPathLength()
    {
        return flagsAndPathLength & kPathLengthBitMask;
    }

    Ray getScatterRay()
    {
        return Ray(this.origin, this.dir, 0, kRayTMax);
    }
}
