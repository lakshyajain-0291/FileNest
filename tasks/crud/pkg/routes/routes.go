package routes

import (
	"crud-api/pkg/controllers"

	"github.com/gin-gonic/gin"
)

func RegisterFileRoutes(router *gin.Engine){
    router.POST("/api/files", controllers.CreateFile)
	router.GET("/api/files", controllers.GetFiles)
    router.GET("/api/files/:id", controllers.GetFileById)
    router.PUT("/api/files/:id", controllers.UpdateFile)
    router.DELETE("/api/files/:id", controllers.DeleteFile)
}