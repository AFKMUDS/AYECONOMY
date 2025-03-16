module ModelAssembly

using Statistics
using Random
using JSON3

export assemble_final_model, export_model, import_model, visualize_model

"""
    assemble_final_model()

Combine all trained knowledge modules into a final model.
"""
function assemble_final_model()
    # Access global modules from TokenSystem
    modules = Main.TokenSystem.modules
    
    # Final compression and refinement
    Main.CompressionSystem.compress_modules(3)  # Lower threshold for final compression
    Main.CompressionSystem.refine_micro_models()
    
    println("\n=== ASSEMBLING FINAL MODEL ===")
    
    # Calculate total tokens and micro-models
    total_tokens = sum(length(module_obj.tokens) for (_, module_obj) in modules)
    total_micro_models = sum(length(module_obj.micro_models) for (_, module_obj) in modules)
    total_references = sum(length(module_obj.references) for (_, module_obj) in modules)
    
    println("Total tokens: $total_tokens")
    println("Total micro-models: $total_micro_models")
    println("Total references: $total_references")
    
    # Show module distribution
    println("\nModule distribution:")
    for (name, module_obj) in sort(collect(modules), by=x->length(x[2].tokens), rev=true)
        token_count = length(module_obj.tokens)
        micro_model_count = length(module_obj.micro_models)
        importance = module_obj.importance_score
        
        println("  $name: $token_count tokens, $micro_model_count micro-models, importance: $(round(importance, digits=2))")
    end
    
    # Show reasoning pathway strengths
    println("\nReasoning pathway strengths:")
    for (source, module_obj) in modules
        for (target, pathways) in module_obj.reasoning_pathways
            if !isempty(pathways)
                println("  $source → $target: $(join(["$(p[1])($(round(p[2], digits=2)))" for p in pathways], ", "))")
            end
        end
    end
    
    return Dict(
        "modules" => modules,
        "stats" => Dict(
            "total_tokens" => total_tokens,
            "total_micro_models" => total_micro_models,
            "total_references" => total_references
        )
    )
end

"""
    export_model(model, output_path::String)

Export the final model to a file.
"""
function export_model(model, output_path::String)
    # Prepare serializable representation
    serializable_model = Dict(
        "modules" => Dict(),
        "stats" => model["stats"]
    )
    
    # Preserve metadata if it exists
    if haskey(model, "metadata")
        serializable_model["metadata"] = model["metadata"]
    end
    
    # Convert modules to serializable format
    for (name, module_obj) in model["modules"]
        serializable_model["modules"][name] = Dict(
            "token_count" => length(module_obj.tokens),
            "reference_count" => length(module_obj.references),
            "micro_model_count" => length(module_obj.micro_models),
            "importance_score" => module_obj.importance_score,
            "transformation_weights" => module_obj.transformation_weights,
            "top_tokens" => []
        )
        
        # Add top tokens
        if !isempty(module_obj.tokens)
            top_tokens = sort(collect(module_obj.tokens), by=x->x[2].importance, rev=true)[1:min(10, length(module_obj.tokens))]
            for (token_str, token_obj) in top_tokens
                token_data = Dict(
                    "value" => token_str,
                    "frequency" => token_obj.frequency,
                    "importance" => token_obj.importance,
                    "connections" => []
                )
                
                # Add top connections
                if !isempty(token_obj.connections)
                    top_connections = sort(collect(token_obj.connections), by=x->x[2], rev=true)[1:min(5, length(token_obj.connections))]
                    for (connected_token, strength) in top_connections
                        push!(token_data["connections"], Dict(
                            "token" => connected_token,
                            "strength" => strength
                        ))
                    end
                end
                
                push!(serializable_model["modules"][name]["top_tokens"], token_data)
            end
        end
    end
    
    # Write to file
    open(output_path, "w") do io
        JSON3.write(io, serializable_model)
    end
    
    println("Exported model to $output_path")
end

"""
    import_model(input_path::String)

Import a model from a file.
"""
function import_model(input_path::String)
    if !isfile(input_path)
        error("Model file not found: $input_path")
    end
    
    # Read the file
    model_data = open(input_path, "r") do io
        JSON3.read(io)
    end
    
    println("Imported model from $input_path")
    println("Model contains $(length(model_data["modules"])) modules")
    
    # Print summary
    println("\nModule summary:")
    for (name, module_data) in model_data["modules"]
        token_count = module_data["token_count"]
        micro_model_count = module_data["micro_model_count"]
        importance = module_data["importance_score"]
        
        println("  $name: $token_count tokens, $micro_model_count micro-models, importance: $(round(importance, digits=2))")
        
        if !isempty(module_data["top_tokens"])
            println("    Top tokens: $(join([t["value"] for t in module_data["top_tokens"][1:min(5, length(module_data["top_tokens"]))]], ", "))")
        end
    end
    
    return model_data
end

"""
    visualize_model(model)

Visualize the model structure.
"""
function visualize_model(model)
    println("\n=== MODEL VISUALIZATION ===")
    
    # Show module distribution
    println("\nModule distribution:")
    for (name, module_data) in sort(collect(model["modules"]), by=x->x[2]["token_count"], rev=true)
        token_count = module_data["token_count"]
        micro_model_count = module_data["micro_model_count"]
        importance = module_data["importance_score"]
        
        println("  $name: $token_count tokens, $micro_model_count micro-models, importance: $(round(importance, digits=2))")
        
        # Show top tokens
        if !isempty(module_data["top_tokens"])
            println("    Top tokens:")
            for (i, token_data) in enumerate(module_data["top_tokens"][1:min(5, length(module_data["top_tokens"]))])
                println("      $i. $(token_data["value"]) (Importance: $(round(token_data["importance"], digits=2)), Frequency: $(token_data["frequency"]))")
                
                # Show connections
                if !isempty(token_data["connections"])
                    connections_str = join(["$(c["token"])($(round(c["strength"], digits=2)))" for c in token_data["connections"]], ", ")
                    println("         → $connections_str")
                end
            end
        end
    end
    
    println("=============================================")
end

end # module ModelAssembly
