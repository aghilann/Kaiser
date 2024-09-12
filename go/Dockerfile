# Use the official Go image as the base
FROM golang:alpine

# Set the working directory to /app
WORKDIR /app

# Copy the Go modules, source code, and main.go
COPY . .

# Download the Go modules
RUN go mod download

# Build the Go binary
RUN go build -o main .

# Expose the port
EXPOSE 8080

# Run the Go API when the container starts
CMD ["./main"]