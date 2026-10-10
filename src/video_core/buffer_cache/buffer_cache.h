// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;
class MemoryTracker;
class PageManager;

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;
    static constexpr u64 STREAM_THRESHOLD = 16_KB;
    /// Up to this much of memory the game keeps writing is copied for each use.
    static constexpr u64 REWRITE_STREAM_THRESHOLD = 512_KB;
    /// At most this much of it is copied a frame, the rest is uploaded.
    static constexpr u64 MaxRewriteCopyBytes = 32_MB;
    /// Uploads are only left to the upload thread up to this size, and while it has less than
    /// this much to copy. When the game loads, gigabytes are uploaded within a frame, and staging
    /// memory can't be used again until the command buffers copying from it are submitted and done,
    /// which they aren't while the thread catches up, so it ran the GPU out of memory.
    static constexpr u64 MaxDeferredUploadBytes = 8_MB;
    static constexpr u64 MaxPendingUploadBytes = 64_MB;
    /// Uploads larger than this are copied through staging memory of this size, a part at a time.
    /// Loading a save in inFAMOUS Second Son, one draw reads 2.4 GB the game wrote all at once,
    /// and staging memory that large couldn't be allocated while other programs held some.
    static constexpr u64 MaxUploadPartBytes = 256_MB;
    /// Buffers read from at least this large are uploaded once until the game could next have
    /// written them for the GPU, see LargeReadSynced.
    static constexpr u64 LARGE_READ_THRESHOLD = 64_MB;

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Makes a scheduler wait, before it submits, for the uploads copied on a thread of their own
    /// that were asked for by then.
    void SetHostWorkOn(Vulkan::Scheduler& target);

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the stream buffer.
    [[nodiscard]] StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    void TickFrame();

    /// Copies back GPU modified memory that game threads read back recently, before they read
    /// it again. Called when the game is signalled that GPU work is done.
    void PrefetchReadbacks();

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool assume_locks = false);

    /// Notes memory written straight to its backing, past the page protection, so the GPU copy
    /// of it is uploaded again before it is used.
    void OnBackingWritten(VAddr device_addr, u64 size);

    /// Notes memory a dispatch of one of the game's compute rings writes, to tell how many of the
    /// copies back game threads wait for are of what those wrote.
    void NoteComputeRingWrite(VAddr device_addr, u64 size) {
        compute_ring_writes[compute_ring_write_index++ % compute_ring_writes.size()] = {
            device_addr, device_addr + size};
    }

    /// Finds a buffer for the specified region. is_read_tracked tells that the caller reports
    /// its accesses to the runtime, which lets small reads use the cached copy in place.
    /// Obtains a buffer for the specified region. When unwritten is given, it is set if the range
    /// returned is known to have nothing written to it since the last barrier, so the caller
    /// needn't find that out again.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(VAddr device_addr, u32 size,
                                                             bool is_written,
                                                             bool is_texel_buffer = false,
                                                             bool is_read_tracked = false,
                                                             bool* unwritten = nullptr);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info);

private:
    using DownloadCopies = boost::container::small_vector<vk::BufferCopy, 8>;

    /// A copy of GPU modified memory back to the game, recorded and submitted by the GPU thread
    /// and waited for by the game thread that touched the memory.
    struct Readback {
        Vulkan::StagingBufferRef staging;
        /// Source offsets are into the arena, destination offsets into the staging buffer.
        DownloadCopies copies;
        VAddr arena_base{};
        VAddr start{};
        VAddr end{};
        /// The copy is done once the semaphore of the queue it is on reaches the tick.
        Vulkan::Semaphore* semaphore{};
        u64 tick{};
        /// Set by the GPU thread when it writes the memory again, so the copy is outdated.
        std::atomic<bool> stale{};
        std::atomic<bool> applied{};
        /// The copy won't be applied, and its ranges are GPU modified again.
        std::atomic<bool> recovered{};
        /// Made ahead of a game thread touching the memory.
        bool prefetched{};
        /// Whether it was counted for or against copying its window ahead. GPU thread.
        bool rated{};
        /// Written back by the thread writing back copies made ahead, rather than by a game
        /// thread waiting for it. Set before applied.
        bool applied_ahead{};
        std::mutex mutex;

        bool Done() const noexcept {
            return applied.load(std::memory_order_acquire) ||
                   recovered.load(std::memory_order_acquire);
        }
    };

    /// Tells the runtime memory is bound to buffers by the next graphics submit.
    void NoteResidencyChange();

    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    ArenaBinds* BindsForArena(const Buffer* arena) {
        NoteResidencyChange();
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }

    const Buffer* GetArena(u64 first_block, u64 last_block);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    /// Returns device memory and an offset into it to back size bytes of arena blocks.
    std::pair<vk::DeviceMemory, u64> AllocateResidency(u64 size);

    /// Returns the arena and the window around a range that is read back with it.
    std::tuple<const Buffer*, VAddr, VAddr> GetReadbackWindow(VAddr device_addr, u64 size);

    /// Records and submits a copy back of the GPU modified memory around a range. GPU thread.
    std::shared_ptr<Readback> StartReadback(VAddr device_addr, u64 size);

    /// Waits for a copy back and writes it to the game's memory. Returns false if it can't be
    /// used as the GPU wrote the memory again. Any thread. Ahead is for the thread writing back
    /// copies made ahead, which waits for the GPU itself and gets false for copies written back
    /// already.
    bool FinishReadback(Readback& readback, bool ahead = false);

    /// Makes the memory of a copy back that won't be used GPU modified again. GPU thread.
    void RecoverReadback(Readback& readback);

    /// Finishes or recovers the copies back overlapping a range. GPU thread.
    void SettleReadbacks(VAddr start, VAddr end);

    /// Frees the staging memory of copies back that are done. GPU thread.
    void PruneReadbacks();

    /// Returns true if memory the game keeps writing should be copied for a draw: once a frame for
    /// each address, and within a budget.
    bool TakeRewriteCopy(VAddr device_addr, u64 size);

    /// Recovers the copies the GPU wrote over again and frees those that are done. GPU thread.
    void ApplyFinishedReadbacks();

    /// Writes back copies made ahead as soon as the GPU is done with them.
    void ReadbackThread(std::stop_token token);

    /// Protects the pages of uploads and copies their memory to staging, in the order the GPU
    /// thread asked for it.
    void UploadThread(std::stop_token token);

    /// Counts a copy made ahead for or against copying its window ahead again. GPU thread.
    void RatePrefetch(Readback& readback, bool useful);

    /// Records a copy back of the GPU modified memory in a window, or returns null if there is
    /// none. GPU thread.
    /// Submits the copies back recorded, on the queues they are on.
    void SubmitReadbacks(bool on_graphics, bool on_async);

    std::shared_ptr<Readback> RecordReadback(const Buffer* arena, VAddr start, VAddr end);

    /// Takes the GPU modified ranges in a range out of the tracked ones, adding copies of them.
    u64 CollectDownloads(const Buffer* arena, VAddr device_addr, u64 size, DownloadCopies& copies);

    void DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    /// Uploads the ranges, guest addresses in dstOffset, MaxUploadPartBytes at a time, waiting
    /// for the GPU to copy each part before the staging memory is filled with the next.
    void UploadInParts(const Buffer* arena, std::span<const vk::BufferCopy> ranges);

    /// Adds a range to the GPU modified ones, unless it is in them already. GPU thread.
    void MarkGpuModifiedRange(VAddr device_addr, u64 size);

    /// Returns true if a large range read from was uploaded since the game last could have
    /// written memory for the GPU to read, and notes that it is about to be if not. GPU thread.
    bool LargeReadSynced(VAddr device_addr, u64 size);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    PageManager& page_manager;
    std::unique_ptr<MemoryTracker> memory_tracker;

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    RangeSet gpu_modified_ranges;
    /// Ranges lately found in or added to gpu_modified_ranges, until some of them is taken out.
    std::array<std::pair<VAddr, VAddr>, 16> gpu_modified_hints{};
    size_t next_gpu_modified_hint{};
    std::vector<std::shared_ptr<Readback>> readbacks;

    /// Windows game threads read back recently, which are copied back ahead.
    struct HotWindow {
        VAddr start;
        VAddr end;
        std::chrono::steady_clock::time_point last_fault;
        /// Chances to copy it ahead skipped after copies of it went stale, and still to skip.
        u8 backoff{};
        u8 skip{};
    };
    std::vector<HotWindow> hot_windows;
    struct ReadbackStats {
        u64 on_fault{};
        u64 joined{};
        u64 prefetched{};
        u64 written_ahead{};
        u64 skipped{};
        /// Time from game threads asking for copies back to the GPU thread taking them up.
        u64 pickup_ns{};
        u64 pickups{};
        /// Command buffers submitted and not done yet when game threads asked, summed.
        u64 in_flight{};
        /// Copies asked for of memory the command buffer being recorded hadn't touched.
        u64 untouched{};
        /// Copies asked for of memory the game's compute rings wrote lately.
        u64 compute_ring{};
    } readback_stats;
    /// The last memory ranges the game's compute rings' dispatches wrote.
    std::array<std::pair<VAddr, VAddr>, 256> compute_ring_writes{};
    size_t compute_ring_write_index{};
    std::chrono::steady_clock::time_point last_readback_report{};

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;
    bool fault_process_pending{};
    struct RewriteCopy {
        VAddr address{};
        u64 frame{};
    };
    /// Addresses of memory the game keeps writing that were copied for a draw, and in which frame.
    std::array<RewriteCopy, 1024> rewrite_copies{};
    u64 rewrite_copy_frame{};
    u64 rewrite_copy_bytes{};
    /// The CPU modified generation all memory in use was last uploaded at for such shaders.
    u64 dma_synced_generation{};
    struct LargeRead {
        VAddr address{};
        u64 size{};
        u64 sync_count{};
    };
    /// Large ranges read from and the Liverpool::sync_count they were last uploaded at.
    std::array<LargeRead, 8> large_reads{};
    /// Whether large ranges read from are uploaded only once in between, see LargeReadSynced.
    bool sync_large_reads_once{true};

    std::array<const Buffer*, NUM_ARENA_PAGES> address_space{};
    std::deque<Buffer> arenas;
    std::vector<ArenaBinds> pending_binds;
    Vulkan::Semaphore memory_semaphore;

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset;
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start)};
        }
    };
    IntervalList<Backing> resident_ranges;
    /// Resident ranges memory was lately found in, in blocks, see EnsureResident.
    std::array<std::pair<u64, u64>, 16> resident_hints{};
    size_t next_resident_hint{};
    vk::DeviceMemory residency_chunk{};
    u64 residency_chunk_used{};
    /// Device memory allocated for residency so far, for the log when an allocation fails.
    u64 residency_allocated_bytes{};

    u32 arena_memory_type_index{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};

    /// Copies made ahead, in the order they were recorded, to be written back once done.
    std::deque<std::shared_ptr<Readback>> finished_readbacks;
    std::mutex finished_readbacks_mutex;
    std::condition_variable_any finished_readbacks_cv;
    /// Declared after what it uses, so it stops before any of that goes away.
    std::jthread readback_thread;

    /// What uploading memory the CPU wrote leaves to a thread of its own: protecting its pages
    /// again, so the CPU's next writes to them are seen, and copying it to staging after that.
    /// The GPU thread spent a tenth of its time on it, most of it in the system changing the
    /// protection. The command buffer the upload is recorded in is only submitted once it is done.
    struct HostUpload {
        PageManager::PendingProtection protects;
        boost::container::small_vector<std::tuple<VAddr, u8*, u64>, 4> copies;
        Vulkan::StagingBufferRef staging{};
    };
    /// Shared with the scheduler, which waits on it to submit, so it outlives the buffer cache.
    struct HostUploads {
        std::mutex mutex;
        std::condition_variable_any cv;
        std::deque<HostUpload> queue;
        std::atomic<u64> queued{};
        std::atomic<u64> done{};
        /// Bytes queued to be copied and not copied yet.
        std::atomic<u64> pending_bytes{};
    };
    std::shared_ptr<HostUploads> host_uploads = std::make_shared<HostUploads>();
    /// Whether uploads are left to that thread.
    bool defer_uploads{true};
    std::jthread upload_thread;
};

} // namespace VideoCore
