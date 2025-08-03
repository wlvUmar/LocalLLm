# Clean up old repo if exists
rm -rf ~/llama.cpp

# Clone the official llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp

# Create and enter build folder
mkdir build && cd build

# Run cmake with necessary flags
cmake .. -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON

# Compile
make -j$(nproc)

# Check if llama-server exists
if [ -f ./bin/llama-server ]; then
    echo "✅ llama-server built successfully!"
    ./bin/llama-server --help | head -n 20
else
    echo "❌ llama-server not found. Build failed."
fi
