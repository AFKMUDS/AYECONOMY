#!/usr/bin/env julia

# Script to run inference on a trained model
include("training_system_modular.jl")

function main()
    if length(ARGS) < 1
        println("Usage: julia run_model.jl \"your query here\"")
        return
    end
    
    model_path = "trained_model.json"
    query = join(ARGS, " ")
    
    println("Loading model from $model_path...")
    
    # Load the model directly as a JSON object instead of using load_model_from_file
    if !isfile(model_path)
        println("Model file not found: $model_path")
        return
    end
    
    try
        # Read the model file directly as JSON
        model_json = JSON3.read(read(model_path, String), Dict{String, Any})
        
        # Initialize modules for inference
        TokenSystem.initialize_modules()
        
        # Run inference with the JSON model
        run_inference(model_json, query)
    catch e
        println("Error: $e")
        return
    end
end

main()
