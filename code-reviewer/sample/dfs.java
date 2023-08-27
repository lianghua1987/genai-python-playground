 //  https://leetcode.com/problems/find-eventual-safe-states/solutions/2063009/java-dfs/

 public List<Integer> eventualSafeNodes(int[][] graph) {
        List<Integer> terminalNodes = new ArrayList<>();
        for (int i = 0; i < graph.length; i++) {
            if (graph[i].length == 0) terminalNodes.add(i);
        }

        List<Integer> safeNodes = new ArrayList<>();
        Boolean[] dp = new Boolean[graph.length];
        for (int i = 0; i < graph.length; i++)
            if (eventualSafeNodes(graph, terminalNodes, i, new boolean[graph.length], dp))
                safeNodes.add(i);
        return safeNodes;
    }

    private boolean eventualSafeNodes(int[][] graph, List<Integer> terminalNodes, int index, boolean[] visited, Boolean[] dp) {
        boolean allMatch = true;
        if (dp[index] != null) return dp[index];
        if (visited[index]) return false;
        visited[index] = true;
        for (Integer target : graph[index]) {
            if (terminalNodes.contains(target)) continue;
            if (!eventualSafeNodes(graph, terminalNodes, target, visited, dp)) {
                allMatch = false;
                break;
            }
        }
        return dp[index] = allMatch;
    }