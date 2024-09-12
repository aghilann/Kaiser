package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

type EchoRequest struct {
    Message string `json:"message" binding:"required"`
}

func HelloHandler(c *gin.Context) {
    c.JSON(http.StatusOK, gin.H{
        "message": "Hello, World!",
    })
}

func EchoHandler(c *gin.Context) {
    var req EchoRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body. 'message' field is required."})
        return
    }
    c.JSON(http.StatusOK, gin.H{"echo": req.Message})
}