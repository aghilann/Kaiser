package models

import (
	"errors"
	"fmt"
)

// Workflow represents the top-level structure
type Workflow struct {
	Workflow             WorkflowInfo         `json:"workflow"`
	Tasks                map[string]TaskInfo  `json:"tasks"`
	Imports              []string             `json:"imports"`
	TopLevelDefinitions  []string             `json:"top_level_definitions"`
	ExecutionID          string               `json:"execution_id"`
}

// WorkflowInfo represents the "workflow" object
type WorkflowInfo struct {
	Name  string         `json:"name"`
	Tasks []WorkflowTask `json:"tasks"`
}

// WorkflowTask represents a task within the workflow
type WorkflowTask struct {
	Task    string          `json:"task"`
	Outputs []string        `json:"outputs"`
	Inputs  [][]interface{} `json:"inputs"`
}

// TaskInfo represents the structure of each task in the "tasks" map
type TaskInfo struct {
	Name           string   `json:"name"`
	Code           string   `json:"code"`
	Inputs         []string `json:"inputs"`
	Outputs        []string `json:"outputs"`
	ContainerImage string   `json:"container_image"`
	CodeToExecute  string   `json:"code_to_execute"`
}

// Graph represents the dependency graph
type Graph struct {
	Adjacency map[string][]string // task -> list of dependent tasks
	InDegree  map[string]int      // task -> number of dependencies
}

// GetGraph builds the dependency graph from the workflow tasks
func (w *Workflow) GetGraph() (*Graph, error) {
	// Step 1: Build outputs map (variable name -> task that produces it)
	outputsMap := make(map[string]string)
	for _, taskInfo := range w.Workflow.Tasks {
		taskName := taskInfo.Task
		for _, output := range taskInfo.Outputs {
			if existingTask, exists := outputsMap[output]; exists {
				return nil, fmt.Errorf("output '%s' is produced by multiple tasks: '%s' and '%s'", output, existingTask, taskName)
			}
			outputsMap[output] = taskName
		}
	}

	// Step 2: Initialize dependencies map (task -> set of tasks it depends on)
	dependencies := make(map[string]map[string]struct{})
	for _, taskInfo := range w.Workflow.Tasks {
		taskName := taskInfo.Task
		dependencies[taskName] = make(map[string]struct{})
	}

	// Step 3: Populate dependencies based on inputs
	for _, taskInfo := range w.Workflow.Tasks {
		taskName := taskInfo.Task
		for _, input := range taskInfo.Inputs {
			if len(input) < 2 {
				return nil, fmt.Errorf("invalid input format for task '%s'", taskName)
			}

			// Extract input variable and its literal flag
			inputVar, ok := input[0].(string)
			if !ok {
				return nil, fmt.Errorf("input variable is not a string for task '%s'", taskName)
			}
			isLiteral, ok := input[1].(bool)
			if !ok {
				return nil, fmt.Errorf("input literal flag is not a boolean for task '%s'", taskName)
			}

			// Only consider inputs that are produced by other tasks (non-literals)
			if !isLiteral {
				producingTask, exists := outputsMap[inputVar]
				if !exists {
					return nil, fmt.Errorf("input variable '%s' for task '%s' is not produced by any task", inputVar, taskName)
				}
				if producingTask != taskName {
					dependencies[taskName][producingTask] = struct{}{}
				}
			}
		}
	}

	// Step 4: Convert dependencies map to adjacency list and calculate in-degrees
	adjacency := make(map[string][]string)
	inDegree := make(map[string]int)

	for task, deps := range dependencies {
		inDegree[task] = len(deps)
		for dep := range deps {
			adjacency[dep] = append(adjacency[dep], task)
		}
	}

	return &Graph{
		Adjacency: adjacency,
		InDegree:  inDegree,
	}, nil
}

// GetTopologicalOrder returns a topological ordering of tasks or an error if not possible
func (w *Workflow) GetTopologicalOrder() ([]string, error) {
	graphResult, err := w.GetGraph()
	if err != nil {
		return nil, err
	}

	graph := graphResult.Adjacency
	inDegree := graphResult.InDegree

	// Initialize queue with tasks having in-degree 0
	queue := []string{}
	for task, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, task)
		}
	}

	var order []string

	// Perform Kahn's algorithm
	for len(queue) > 0 {
		// Dequeue
		current := queue[0]
		queue = queue[1:]
		order = append(order, current)

		// Decrease in-degree of neighboring tasks
		for _, neighbor := range graph[current] {
			inDegree[neighbor]--
			if inDegree[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}

	// Check for cycles
	if len(order) != len(w.Tasks) {
		return nil, errors.New("workflow contains a cyclic dependency")
	}

	return order, nil
}