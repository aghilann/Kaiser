package handlers

import (
	"net/http"

	"github.com/aghilann/Kaiser/go/src/models"
	"github.com/gin-gonic/gin"
)

func ScheduleWorkflow(c *gin.Context) {
    var workflow models.Workflow
    if err := c.ShouldBindJSON(&workflow); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid workflow serialization"})
        return
    }
    c.JSON(http.StatusOK, gin.H{"echo": workflow.ExecutionID})
}

func GetTopologicalOrder(c *gin.Context) {
	var workflow models.Workflow

	if err := c.ShouldBindJSON(&workflow); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid workflow serialization"})
        return
	}

	topologicalOrder, err := workflow.GetTopologicalOrder()
	if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err})
        return
    }

	c.JSON(http.StatusOK, gin.H{"topological_order": topologicalOrder})
}