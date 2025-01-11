# Kotlin Problem-Solving Cheat Sheet

## Index
1. [Arrays and Strings](#1-arrays-and-strings)
2. [Linked Lists](#2-linked-lists)
3. [Stacks and Queues](#3-stacks-and-queues)
4. [Trees and Graphs](#4-trees-and-graphs)
5. [Dynamic Programming (DP)](#5-dynamic-programming-dp)
6. [Sorting and Searching](#6-sorting-and-searching)
7. [Hashing](#7-hashing)
8. [Heaps](#8-heaps)
9. [Backtracking](#9-backtracking)
10. [Bit Manipulation](#10-bit-manipulation)
11. [Summary of When to Use Each Approach](#summary-of-when-to-use-each-approach)

## 1. Arrays and Strings

### Two-Pointer Technique
Useful for problems involving searching pairs in sorted arrays or strings.

**Example:** Finding two numbers in a sorted array that sum up to a target value.

```kotlin
fun twoSum(nums: IntArray, target: Int): IntArray {
    var left = 0
    var right = nums.size - 1
    while (left < right) {
        val sum = nums[left] + nums[right]
        when {
            sum == target -> return intArrayOf(left, right)
            sum < target -> left++
            else -> right--
        }
    }
    return intArrayOf()
}
```

### Sliding Window
Ideal for problems involving subarrays or substrings.

**Example:** Finding the maximum sum of a subarray with a fixed size.

```kotlin
fun maxSumSubarray(nums: IntArray, k: Int): Int {
    var maxSum = 0
    var windowSum = 0
    for (i in nums.indices) {
        windowSum += nums[i]
        if (i >= k - 1) {
            maxSum = maxOf(maxSum, windowSum)
            windowSum -= nums[i - (k - 1)]
        }
    }
    return maxSum
}
```

## 2. Linked Lists

### Fast and Slow Pointers
Effective for cycle detection and finding the middle element.

**Example:** Detecting a cycle in a linked list.

```kotlin
class ListNode(var value: Int) {
    var next: ListNode? = null
}

fun hasCycle(head: ListNode?): Boolean {
    var slow = head
    var fast = head
    while (fast?.next != null) {
        slow = slow?.next
        fast = fast.next?.next
        if (slow == fast) return true
    }
    return false
}
```

### Dummy Nodes
Simplifies edge cases in insertion and deletion operations.

**Example:** Removing the nth node from the end of a linked list.

```kotlin
fun removeNthFromEnd(head: ListNode?, n: Int): ListNode? {
    val dummy = ListNode(0)
    dummy.next = head
    var first: ListNode? = dummy
    var second: ListNode? = dummy
    for (i in 0..n) {
        first = first?.next
    }
    while (first != null) {
        first = first.next
        second = second?.next
    }
    second?.next = second?.next?.next
    return dummy.next
}
```

## 3. Stacks and Queues

### Monotonic Stack
Helps in problems requiring the next greater or smaller element.

**Example:** Finding the next greater element for each element in an array.

```kotlin
fun nextGreaterElements(nums: IntArray): IntArray {
    val result = IntArray(nums.size) { -1 }
    val stack = mutableListOf<Int>()
    for (i in nums.indices) {
        while (stack.isNotEmpty() && nums[stack.last()] < nums[i]) {
            result[stack.removeAt(stack.size - 1)] = nums[i]
        }
        stack.add(i)
    }
    return result
}
```

### Queue with BFS
Essential for level-order traversal in trees or shortest path in unweighted graphs.

**Example:** Performing level-order traversal of a binary tree.

```kotlin
fun levelOrder(root: TreeNode?): List<List<Int>> {
    val result = mutableListOf<List<Int>>>()
    if (root == null) return result
    val queue: Queue<TreeNode> = LinkedList()
    queue.add(root)
    while (queue.isNotEmpty()) {
        val level = mutableListOf<Int>()
        val size = queue.size
        for (i in 0 until size) {
            val node = queue.poll()
            level.add(node.`val`)
            node.left?.let { queue.add(it) }
            node.right?.let { queue.add(it) }
        }
        result.add(level)
    }
    return result
}
```

## 4. Trees and Graphs

### Depth-First Search (DFS)
Explores all nodes in a branch before backtracking.

**Example:** Checking if a binary tree is symmetric.

```kotlin
fun isSymmetric(root: TreeNode?): Boolean {
    fun isMirror(t1: TreeNode?, t2: TreeNode?): Boolean {
        if (t1 == null && t2 == null) return true
        if (t1 == null || t2 == null) return false
        return (t1.`val` == t2.`val`)
            && isMirror(t1.left, t2.right)
            && isMirror(t1.right, t2.left)
    }
    return isMirror(root, root)
}
```

### Breadth-First Search (BFS)
Explores all neighbors at the present depth before moving on.

**Example:** Finding the shortest path in an unweighted graph.

```kotlin
fun shortestPath(graph: List<List<Int>>, start: Int, end: Int): Int {
    val queue: Queue<Pair<Int, Int>> = LinkedList()
    val visited = mutableSetOf<Int>()
    queue.add(Pair(start, 0))
    visited.add(start)
    while (queue.isNotEmpty()) {
        val (node, depth) = queue.poll()
        if (node == end) return depth
        for (neighbor in graph[node]) {
            if (neighbor !in visited) {
                visited.add(neighbor)
                queue.add(Pair(neighbor, depth + 1))
            }
        }
    }
    return -1 // Path not found
}
```

### Union-Find
Manages disjoint sets efficiently.

**Example:** Detecting cycles in an undirected graph.

```kotlin
class UnionFind(size: Int) {
    private val parent = IntArray(size) { it }
    private val rank = IntArray(size)
}
```

## 5. Dynamic Programming (DP)

### Memoization (Top-Down Approach)
When to Use: Optimizing recursive solutions with overlapping subproblems.

**Example:** Fibonacci sequence using memoization.

```kotlin
fun fib(n: Int, memo: MutableMap<Int, Int> = mutableMapOf()): Int {
    if (n <= 1) return n
    if (memo.containsKey(n)) return memo[n]!!
    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
    return memo[n]!!
}
```

### Tabulation (Bottom-Up Approach)
When to Use: Iterative solutions like Knapsack, Longest Increasing Subsequence.

**Example:** Fibonacci sequence using tabulation.

```kotlin
fun fibTabulation(n: Int): Int {
    if (n <= 1) return n
    val dp = IntArray(n + 1)
    dp[0] = 0
    dp[1] = 1
    for (i in 2..n) {
        dp[i] = dp[i - 1] + dp[i - 2]
    }
    return dp[n]
}
```

## 6. Sorting and Searching

### Binary Search
When to Use: Finding target values in sorted arrays.

**Example:** Binary search in a sorted array.

```kotlin
fun binarySearch(nums: IntArray, target: Int): Int {
    var left = 0
    var right = nums.size - 1
    while (left <= right) {
        val mid = left + (right - left) / 2
        when {
            nums[mid] == target -> return mid
            nums[mid] < target -> left = mid + 1
            else -> right = mid - 1
        }
    }
    return -1
}
```

### Merge Sort
When to Use: Efficient sorting of large datasets.

**Example:** Merge Sort.

```kotlin
fun mergeSort(arr: IntArray): IntArray {
    if (arr.size <= 1) return arr
    val mid = arr.size / 2
    val left = mergeSort(arr.sliceArray(0 until mid))
    val right = mergeSort(arr.sliceArray(mid until arr.size))
    return merge(left, right)
}

fun merge(left: IntArray, right: IntArray): IntArray {
    var i = 0
    var j = 0
    val merged = mutableListOf<Int>()
    while (i < left.size && j < right.size) {
        if (left[i] <= right[j]) merged.add(left[i++]) else merged.add(right[j++])
    }
    merged.addAll(left.slice(i until left.size))
    merged.addAll(right.slice(j until right.size))
    return merged.toIntArray()
}
```

## 7. Hashing

### Hash Maps/Sets
When to Use: Frequency counting, duplicate detection, caching.

**Example:** Checking for duplicates in an array.

```kotlin
fun containsDuplicate(nums: IntArray): Boolean {
    val seen = mutableSetOf<Int>()
    for (num in nums) {
        if (!seen.add(num)) return true
    }
    return false
}
```

## 8. Heaps

### Min-Heap/Max-Heap
When to Use: Priority queues, finding the k-th largest/smallest element.

**Example:** Finding the k-th largest element in an array.

```kotlin
import java.util.PriorityQueue

fun findKthLargest(nums: IntArray, k: Int): Int {
    val minHeap = PriorityQueue<Int>()
    for (num in nums) {
        minHeap.add(num)
        if (minHeap.size > k) minHeap.poll()
    }
    return minHeap.peek()
}
```

## 9. Backtracking

### Recursive Search
When to Use: Permutations, combinations, puzzles (Sudoku, N-Queens).

**Example:** Generating all permutations of an array.

```kotlin
fun permute(nums: IntArray): List<List<Int>> {
    val result = mutableListOf<List<Int>>()
    fun backtrack(path: MutableList<Int>, used: BooleanArray) {
        if (path.size == nums.size) {
            result.add(ArrayList(path))
            return
        }
        for (i in nums.indices) {
            if (used[i]) continue
            used[i] = true
            path.add(nums[i])
            backtrack(path, used)
            path.removeAt(path.size - 1)
            used[i] = false
        }
    }
    backtrack(mutableListOf(), BooleanArray(nums.size))
    return result
}
```

## 10. Bit Manipulation

### Bitwise Operations
When to Use: Checking even/odd, toggling bits, counting set bits.

**Example:** Counting the number of set bits (Hamming Weight).

```kotlin
fun hammingWeight(n: Int): Int {
    var count = 0
    var num = n
    while (num != 0) {
        count += num and 1
        num = num ushr 1
    }
    return count
}
```

### Checking if a Number is a Power of Two

```kotlin
fun isPowerOfTwo(n: Int): Boolean {
    return n > 0 && (n and (n - 1)) == 0
}
```

## Summary of When to Use Each Approach

| Technique             | When to Use                                                   |
|-----------------------|---------------------------------------------------------------|
| Memoization (DP)      | Recursive problems with overlapping subproblems (Fibonacci, Grid Paths) |
| Tabulation (DP)       | Iterative problems (Knapsack, Longest Increasing Subsequence) |
| Binary Search         | Sorted arrays for searching/insertion                         |
| Merge/Quick Sort      | Sorting large datasets                                        |
| Hash Maps/Sets        | Frequency counting, duplicate detection                       |
| Heaps (Priority Queues)| K-th largest/smallest, task scheduling                        |
| Backtracking          | Combinations, permutations, constraint satisfaction (Sudoku)  |
| Bit Manipulation      | Optimized operations for state toggling, parity checking      |
