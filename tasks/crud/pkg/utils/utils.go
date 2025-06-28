package utils

import "github.com/gin-gonic/gin"

// ThrowError sends a JSON error response with the provided status code, error, and context message.
// If err is nil, it returns "nil" as the error message.
func ThrowError(c *gin.Context, err error, status int, Context string){
	if(err != nil){
		c.JSON(status, gin.H{"Error: ": err.Error(), "Context: ": Context})
	} else {
		c.JSON(status, gin.H{"Error: ": "nil", "Context: ": Context})
	}
	
}