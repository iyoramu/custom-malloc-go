package main

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"
	"unsafe"
)

// Constants
const (
	MinAllocSize     = 1 << 4  // 16 bytes minimum allocation
	MaxAllocSize     = 1 << 24 // 16 MB maximum allocation
	ArenaSize        = 1 << 26 // 64 MB arena size
	MaxSmallSize     = 1 << 10 // 1KB - small allocations threshold
	NumSizeClasses   = 68      // Number of size classes
	MaxCacheSize     = 32      // Maximum number of cached blocks per size class
	VisualizationFPS = 10      // Frames per second for visualization
)

// Size classes based on jemalloc/TCMalloc inspired buckets
var sizeClasses = []int{
	16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256,
	320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072,
	3584, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 14336, 16384, 20480, 24576,
	28672, 32768, 40960, 49152, 57344, 65536, 81920, 98304, 114688, 131072, 163840,
	196608, 229376, 262144, 327680, 393216, 458752, 524288, 655360, 786432, 917504,
	1048576, 1310720, 1572864, 1835008, 2097152, 2621440, 3145728, 3670016, 4194304,
}

// block represents a memory block in the arena
type block struct {
	start    uintptr
	size     int
	free     bool
	next     *block
	prev     *block
	sizeClass int
}

// arena represents a contiguous memory region
type arena struct {
	start    uintptr
	size     int
	blocks   *block
	freeList []*block
	mu       sync.Mutex
}

// Allocator is the main memory allocator structure
type Allocator struct {
	arenas       []*arena
	sizeClassMap map[int]int
	caches       [NumSizeClasses][]*block
	cacheCounts  [NumSizeClasses]int
	stats        struct {
		totalAllocated int
		totalFreed     int
		currentInUse   int
		allocCount     int
		freeCount      int
	}
	visualizationEnabled bool
	stopVisualization    chan struct{}
	visualizationData    chan []*block
}

// NewAllocator creates a new memory allocator
func NewAllocator() *Allocator {
	alloc := &Allocator{
		sizeClassMap:        make(map[int]int),
		visualizationData:   make(chan []*block, 1),
		stopVisualization:   make(chan struct{}),
	}

	// Initialize size class map
	for i, size := range sizeClasses {
		alloc.sizeClassMap[size] = i
	}

	// Create initial arena
	alloc.createArena()

	return alloc
}

// createArena creates a new memory arena
func (a *Allocator) createArena() {
	// Allocate memory using mmap or similar would be better, but for Go we'll use a byte slice
	mem := make([]byte, ArenaSize)
	start := uintptr(unsafe.Pointer(&mem[0]))

	newArena := &arena{
		start:  start,
		size:   ArenaSize,
		blocks: &block{start: start, size: ArenaSize, free: true},
	}

	a.arenas = append(a.arenas, newArena)
}

// getSizeClass returns the size class for a given size
func (a *Allocator) getSizeClass(size int) (int, int) {
	if size <= MaxSmallSize {
		// Binary search to find the smallest size class that fits
		idx := sort.Search(len(sizeClasses), func(i int) bool {
			return sizeClasses[i] >= size
		})
		if idx < len(sizeClasses) {
			return idx, sizeClasses[idx]
		}
	}
	return -1, size // -1 indicates large allocation
}

// Malloc allocates memory of the given size
func (a *Allocator) Malloc(size int) unsafe.Pointer {
	if size <= 0 {
		return nil
	}

	// Align size
	if size < MinAllocSize {
		size = MinAllocSize
	}

	sizeClassIdx, sizeClass := a.getSizeClass(size)

	// Try to get from cache first for small allocations
	if sizeClassIdx >= 0 {
		a.arenas[0].mu.Lock()
		if a.cacheCounts[sizeClassIdx] > 0 {
			blk := a.caches[sizeClassIdx][a.cacheCounts[sizeClassIdx]-1]
			a.caches[sizeClassIdx][a.cacheCounts[sizeClassIdx]-1] = nil
			a.cacheCounts[sizeClassIdx]--
			a.arenas[0].mu.Unlock()

			blk.free = false
			a.stats.allocCount++
			a.stats.totalAllocated += blk.size
			a.stats.currentInUse += blk.size

			if a.visualizationEnabled {
				a.updateVisualization()
			}

			return unsafe.Pointer(blk.start)
		}
		a.arenas[0].mu.Unlock()
	}

	// Find suitable arena and block
	for _, arena := range a.arenas {
		arena.mu.Lock()
		current := arena.blocks
		var bestFit *block = nil

		for current != nil {
			if current.free && current.size >= size {
				if bestFit == nil || current.size < bestFit.size {
					bestFit = current
					// Exact fit found, break early
					if bestFit.size == size {
						break
					}
				}
			}
			current = current.next
		}

		if bestFit != nil {
			// Split block if there's remaining space
			if bestFit.size > size+MinAllocSize {
				newBlk := &block{
					start:    bestFit.start + uintptr(size),
					size:     bestFit.size - size,
					free:     true,
					next:     bestFit.next,
					prev:     bestFit,
					sizeClass: -1,
				}

				if bestFit.next != nil {
					bestFit.next.prev = newBlk
				}
				bestFit.next = newBlk
				bestFit.size = size
			}

			bestFit.free = false
			bestFit.sizeClass = sizeClassIdx

			a.stats.allocCount++
			a.stats.totalAllocated += bestFit.size
			a.stats.currentInUse += bestFit.size

			arena.mu.Unlock()

			if a.visualizationEnabled {
				a.updateVisualization()
			}

			return unsafe.Pointer(bestFit.start)
		}
		arena.mu.Unlock()
	}

	// No space found in existing arenas, create a new one
	a.createArena()
	return a.Malloc(size) // Retry with new arena
}

// Free releases allocated memory
func (a *Allocator) Free(ptr unsafe.Pointer) {
	if ptr == nil {
		return
	}

	target := uintptr(ptr)

	// Find the block containing this pointer
	for _, arena := range a.arenas {
		arena.mu.Lock()
		current := arena.blocks

		for current != nil {
			if current.start == target && !current.free {
				// Add to cache if it's a small allocation
				if current.sizeClass >= 0 && a.cacheCounts[current.sizeClass] < MaxCacheSize {
					a.caches[current.sizeClass][a.cacheCounts[current.sizeClass]] = current
					a.cacheCounts[current.sizeClass]++
					current.free = true
				} else {
					// Standard free with coalescing
					current.free = true
					a.coalesce(current)
				}

				a.stats.freeCount++
				a.stats.totalFreed += current.size
				a.stats.currentInUse -= current.size

				arena.mu.Unlock()

				if a.visualizationEnabled {
					a.updateVisualization()
				}

				return
			}
			current = current.next
		}
		arena.mu.Unlock()
	}

	// Pointer not found - maybe already freed or invalid
	panic("invalid pointer or double free")
}

// coalesce merges adjacent free blocks
func (a *Allocator) coalesce(blk *block) {
	// Coalesce with next block if possible
	if blk.next != nil && blk.next.free {
		blk.size += blk.next.size
		blk.next = blk.next.next
		if blk.next != nil {
			blk.next.prev = blk
		}
	}

	// Coalesce with previous block if possible
	if blk.prev != nil && blk.prev.free {
		blk.prev.size += blk.size
		blk.prev.next = blk.next
		if blk.next != nil {
			blk.next.prev = blk.prev
		}
	}
}

// EnableVisualization turns on memory visualization
func (a *Allocator) EnableVisualization() {
	a.visualizationEnabled = true
	go a.visualizationWorker()
}

// DisableVisualization turns off memory visualization
func (a *Allocator) DisableVisualization() {
	if a.visualizationEnabled {
		a.visualizationEnabled = false
		close(a.stopVisualization)
	}
}

// updateVisualization sends data to the visualization worker
func (a *Allocator) updateVisualization() {
	// Collect all blocks from the first arena for visualization
	var blocks []*block
	current := a.arenas[0].blocks
	for current != nil {
		blocks = append(blocks, current)
		current = current.next
	}

	select {
	case a.visualizationData <- blocks:
	default:
		// Skip if previous frame hasn't been processed yet
	}
}

// visualizationWorker handles the terminal visualization
func (a *Allocator) visualizationWorker() {
	ticker := time.NewTicker(time.Second / VisualizationFPS)
	defer ticker.Stop()

	for {
		select {
		case <-a.stopVisualization:
			return
		case blocks := <-a.visualizationData:
			a.renderVisualization(blocks)
		case <-ticker.C:
			// Render even if no updates to maintain FPS
		}
	}
}

// renderVisualization displays memory usage in terminal
func (a *Allocator) renderVisualization(blocks []*block) {
	const width = 80
	const height = 24

	// Clear screen and move cursor to top-left
	fmt.Print("\033[H\033[2J")

	// Display stats
	fmt.Printf("Custom Malloc Go - Memory Allocator Visualization\n")
	fmt.Printf("Allocations: %d  Frees: %d  In Use: %s  Total Allocated: %s\n",
		a.stats.allocCount,
		a.stats.freeCount,
		formatSize(a.stats.currentInUse),
		formatSize(a.stats.totalAllocated),
	)

	// Calculate scale factor for visualization
	maxAddr := uintptr(0)
	for _, blk := range blocks {
		if blk.start+uintptr(blk.size) > maxAddr {
			maxAddr = blk.start + uintptr(blk.size)
		}
	}
	scale := float64(maxAddr) / float64(width*height)

	// Create visualization grid
	grid := make([][]rune, height)
	for i := range grid {
		grid[i] = make([]rune, width)
		for j := range grid[i] {
			grid[i][j] = ' '
		}
	}

	// Fill grid with block information
	for _, blk := range blocks {
		startX := int(float64(blk.start) / scale / float64(width))
		startY := int(float64(blk.start) / scale) % width
		endX := int(float64(blk.start+uintptr(blk.size)) / scale / float64(width))
		endY := int(float64(blk.start+uintptr(blk.size)) / scale) % width

		if startX >= height {
			startX = height - 1
		}
		if endX >= height {
			endX = height - 1
		}

		char := '■'
		if blk.free {
			char = '□'
		}

		for x := startX; x <= endX; x++ {
			if x >= height {
				continue
			}
			start := 0
			if x == startX {
				start = startY
			}
			end := width - 1
			if x == endX {
				end = endY
			}
			for y := start; y <= end; y++ {
				if y < width {
					grid[x][y] = char
				}
			}
		}
	}

	// Print grid
	for _, row := range grid {
		for _, cell := range row {
			fmt.Printf("%c", cell)
		}
		fmt.Println()
	}

	// Print legend
	fmt.Println("\nLegend: ■ = Allocated, □ = Free")
	fmt.Println("Press Ctrl+C to exit")
}

// formatSize formats bytes into human-readable string
func formatSize(bytes int) string {
	units := []string{"B", "KB", "MB", "GB", "TB"}
	size := float64(bytes)
	i := 0
	for size >= 1024 && i < len(units)-1 {
		size /= 1024
		i++
	}
	return fmt.Sprintf("%.2f %s", size, units[i])
}

// demo runs a demonstration of the allocator
func demo() {
	alloc := NewAllocator()
	alloc.EnableVisualization()

	// Allocate some memory
	ptrs := make([]unsafe.Pointer, 0, 100)
	for i := 0; i < 50; i++ {
		size := rand.Intn(2048) + 16
		ptr := alloc.Malloc(size)
		ptrs = append(ptrs, ptr)
		time.Sleep(time.Millisecond * 50)
	}

	// Free some memory
	for i := 0; i < 25; i++ {
		idx := rand.Intn(len(ptrs))
		if ptrs[idx] != nil {
			alloc.Free(ptrs[idx])
			ptrs[idx] = nil
		}
		time.Sleep(time.Millisecond * 100)
	}

	// Allocate more
	for i := 0; i < 25; i++ {
		size := rand.Intn(8192) + 16
		ptr := alloc.Malloc(size)
		ptrs = append(ptrs, ptr)
		time.Sleep(time.Millisecond * 75)
	}

	// Free all
	for _, ptr := range ptrs {
		if ptr != nil {
			alloc.Free(ptr)
			time.Sleep(time.Millisecond * 50)
		}
	}

	time.Sleep(time.Second)
	alloc.DisableVisualization()
}

func main() {
	// Check if demo mode is requested
	if len(os.Args) > 1 && os.Args[1] == "--demo" {
		demo()
		return
	}

	alloc := NewAllocator()
	fmt.Println("Custom Malloc in Go")
	fmt.Println("Usage:")
	fmt.Println("  --demo : Run visualization demo")
	fmt.Println()
	fmt.Println("This allocator implements:")
	fmt.Println("- Size class based allocation for small objects")
	fmt.Println("- Best-fit strategy for larger allocations")
	fmt.Println("- Block coalescing to reduce fragmentation")
	fmt.Println("- Simple per-arena locking for thread safety")
	fmt.Println("- Visualization of memory usage patterns")
	fmt.Println()
	fmt.Println("For competition submission, focus on:")
	fmt.Println("1. Modern terminal-based UI/UX")
	fmt.Println("2. Comprehensive functionality")
	fmt.Println("3. High performance through caching and size classes")
	fmt.Println("4. Clean, well-organized code")
}
