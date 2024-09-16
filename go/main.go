package main

import (
	"github.com/aghilann/Kaiser/go/src/handlers"
	"github.com/gin-gonic/gin"
)

func main() {
    r := gin.Default()

    r.GET("/hello", handlers.HelloHandler)
    r.POST("/echo", handlers.EchoHandler)
    r.POST("/workflow", handlers.ScheduleWorkflow)
    r.POST("/workflow/topological-order", handlers.GetTopologicalOrder)

    r.Run(":8080")
}