module CompressionSystem

using Statistics
using Random

export compress_modules, refine_micro_models, prune_low_importance_tokens

"""
    compress_modules(frequency_threshold::Int=5; thorough::Bool=false)

Compress tokens by replacing duplicates with references and pruning low-importance tokens.
When thorough=true, performs a more aggressive compression with similarity detection.
"""
function compress_modules(frequency_threshold::Int=5; thorough::Bool=false)
    # Access global modules from TokenSystem
    modules = Main.TokenSystem.modules
    
    total_tokens = 0
    total_references = 0
    tokens_removed = 0
    similar_tokens_merged = 0
    
    # Adjust thresholds for thorough compression
    importance_threshold = thorough ? 0.5 : 0.3
    frequency_min = thorough ? 2 : 3
    
    for (name, module_obj) in modules
        # Find high-frequency tokens
        high_value_tokens = filter(p -> p.second.frequency > frequency_threshold, module_obj.tokens)
        
        # Create references for these tokens
        for (token_str, token_obj) in high_value_tokens
            ref_id = "ref_$(name)_$(hash(token_str))"
            module_obj.references[token_str] = ref_id
        end
        
        # Count tokens and references
        total_tokens += length(module_obj.tokens)
        total_references += length(module_obj.references)
        
        # Prune low-importance tokens to save memory
        low_importance_tokens = filter(p -> p.second.importance < importance_threshold && p.second.frequency < frequency_min, module_obj.tokens)
        for (token_str, _) in low_importance_tokens
            delete!(module_obj.tokens, token_str)
            tokens_removed += 1
        end
        
        # For thorough compression, also merge similar tokens
        if thorough
            # Get all token pairs
            token_pairs = collect(keys(module_obj.tokens))
            
            # Check each pair for similarity
            for i in 1:length(token_pairs)
                for j in (i+1):length(token_pairs)
                    if j <= length(token_pairs) && haskey(module_obj.tokens, token_pairs[i]) && haskey(module_obj.tokens, token_pairs[j])
                        token1 = module_obj.tokens[token_pairs[i]]
                        token2 = module_obj.tokens[token_pairs[j]]
                        
                        # Calculate similarity between tokens
                        similarity = calculate_token_similarity(token1, token2)
                        
                        # If tokens are very similar, merge them
                        if similarity > 0.85
                            # Keep the token with higher importance
                            if token1.importance >= token2.importance
                                # Merge token2 into token1
                                token1.frequency += token2.frequency
                                token1.importance = max(token1.importance, token2.importance) * 1.1
                                
                                # Merge connections
                                for (conn_token, strength) in token2.connections
                                    if haskey(token1.connections, conn_token)
                                        token1.connections[conn_token] = max(token1.connections[conn_token], strength)
                                    else
                                        token1.connections[conn_token] = strength
                                    end
                                end
                                
                                # Remove token2
                                delete!(module_obj.tokens, token_pairs[j])
                                similar_tokens_merged += 1
                            else
                                # Merge token1 into token2
                                token2.frequency += token1.frequency
                                token2.importance = max(token2.importance, token1.importance) * 1.1
                                
                                # Merge connections
                                for (conn_token, strength) in token1.connections
                                    if haskey(token2.connections, conn_token)
                                        token2.connections[conn_token] = max(token2.connections[conn_token], strength)
                                    else
                                        token2.connections[conn_token] = strength
                                    end
                                end
                                
                                # Remove token1
                                delete!(module_obj.tokens, token_pairs[i])
                                similar_tokens_merged += 1
                                break  # Break inner loop as token1 is now gone
                            end
                        end
                    end
                end
            end
        end
    end
    
    println("Compression stats: $total_tokens tokens, $total_references references")
    println("Compression ratio: $(round(total_references/max(1,total_tokens), digits=2))")
    println("Pruned $tokens_removed low-importance tokens")
    
    if thorough
        println("Merged $similar_tokens_merged similar tokens")
        
        # Also update reasoning pathways
        if isdefined(Main.TokenSystem, :update_reasoning_pathways)
            println("Updating reasoning pathways...")
            Main.TokenSystem.update_reasoning_pathways()
        else
            println("Warning: update_reasoning_pathways not defined, skipping pathway updates")
        end
    end
    
    # Force garbage collection to free memory
    GC.gc()
end

"""
    refine_micro_models()

Identify and refine micro-models within each knowledge module.
"""
function refine_micro_models()
    # Access global modules from TokenSystem
    modules = Main.TokenSystem.modules
    
    for (name, module_obj) in modules
        # Skip if no tokens
        if isempty(module_obj.tokens)
            continue
        end
        
        # Find strongly connected token groups
        token_graph = Dict{String, Vector{Tuple{String, Float64}}}()
        
        # Build graph representation
        for (token_str, token_obj) in module_obj.tokens
            if !isempty(token_obj.connections)
                token_graph[token_str] = [(next, strength) for (next, strength) in token_obj.connections]
            end
        end
        
        # Find connected components (simple approach)
        if !isempty(token_graph)
            # Only keep the top 5 micro-models per module to avoid memory issues
            module_obj.micro_models = module_obj.micro_models[max(1, length(module_obj.micro_models)-5):end]
        end
    end
end

"""
    merge_similar_tokens(similarity_threshold::Float64=0.8)

Merge tokens that are semantically similar to reduce redundancy.
"""
function merge_similar_tokens(similarity_threshold::Float64=0.8)
    # Access global modules from TokenSystem
    modules = Main.TokenSystem.modules
    
    total_merges = 0
    
    for (name, module_obj) in modules
        # Skip if too few tokens
        if length(module_obj.tokens) < 2
            continue
        end
        
        # Get all token pairs
        token_pairs = []
        token_keys = collect(keys(module_obj.tokens))
        
        for i in 1:(length(token_keys)-1)
            for j in (i+1):length(token_keys)
                push!(token_pairs, (token_keys[i], token_keys[j]))
            end
        end
        
        # Check similarity for each pair
        for (token1, token2) in token_pairs
            # Skip if either token has been deleted
            if !haskey(module_obj.tokens, token1) || !haskey(module_obj.tokens, token2)
                continue
            end
            
            # Calculate similarity (simplified)
            # In a real system, we would use more sophisticated similarity measures
            similarity = calculate_token_similarity(module_obj.tokens[token1], module_obj.tokens[token2])
            
            if similarity > similarity_threshold
                # Merge tokens
                merge_tokens(module_obj, token1, token2)
                total_merges += 1
            end
        end
    end
    
    println("Merged $total_merges similar tokens")
end

"""
    calculate_token_similarity(token1::Main.TokenSystem.Token, token2::Main.TokenSystem.Token)

Calculate similarity between two tokens based on their connections.
"""
function calculate_token_similarity(token1::Main.TokenSystem.Token, token2::Main.TokenSystem.Token)
    # If both tokens have no connections, they're not similar
    if isempty(token1.connections) && isempty(token2.connections)
        return 0.0
    end
    
    # Get common connections
    common_connections = intersect(keys(token1.connections), keys(token2.connections))
    
    # Calculate Jaccard similarity
    union_connections = union(keys(token1.connections), keys(token2.connections))
    
    if isempty(union_connections)
        return 0.0
    end
    
    return length(common_connections) / length(union_connections)
end

"""
    merge_tokens(module_obj::Main.TokenSystem.KnowledgeModule, token1::String, token2::String)

Merge two tokens, keeping the one with higher importance.
"""
function merge_tokens(module_obj::Main.TokenSystem.KnowledgeModule, token1::String, token2::String)
    # Determine which token to keep (the one with higher importance)
    keep_token, remove_token = if module_obj.tokens[token1].importance > module_obj.tokens[token2].importance
        (token1, token2)
    else
        (token2, token1)
    end
    
    # Merge frequency and importance
    module_obj.tokens[keep_token].frequency += module_obj.tokens[remove_token].frequency
    module_obj.tokens[keep_token].importance = max(
        module_obj.tokens[keep_token].importance,
        module_obj.tokens[remove_token].importance
    )
    
    # Merge connections
    for (connected_token, strength) in module_obj.tokens[remove_token].connections
        if haskey(module_obj.tokens[keep_token].connections, connected_token)
            # Take the stronger connection
            module_obj.tokens[keep_token].connections[connected_token] = max(
                module_obj.tokens[keep_token].connections[connected_token],
                strength
            )
        else
            # Add new connection
            module_obj.tokens[keep_token].connections[connected_token] = strength
        end
    end
    
    # Update any connections that point to the removed token
    for (token_str, token_obj) in module_obj.tokens
        if haskey(token_obj.connections, remove_token)
            strength = token_obj.connections[remove_token]
            delete!(token_obj.connections, remove_token)
            token_obj.connections[keep_token] = get(token_obj.connections, keep_token, 0.0) + strength
        end
    end
    
    # Remove the merged token
    delete!(module_obj.tokens, remove_token)
    
    # Update micro-models
    for i in 1:length(module_obj.micro_models)
        module_obj.micro_models[i] = [token == remove_token ? keep_token : token for token in module_obj.micro_models[i]]
    end
end

"""
    prune_weak_connections(strength_threshold::Float64=0.2)

Remove weak connections between tokens to reduce noise.
"""
function prune_weak_connections(strength_threshold::Float64=0.2)
    # Access global modules from TokenSystem
    modules = Main.TokenSystem.modules
    
    total_pruned = 0
    
    for (name, module_obj) in modules
        for (token_str, token_obj) in module_obj.tokens
            # Find weak connections
            weak_connections = filter(p -> p.second < strength_threshold, token_obj.connections)
            
            # Remove weak connections
            for (connected_token, _) in weak_connections
                delete!(token_obj.connections, connected_token)
                total_pruned += 1
            end
        end
    end
    
    println("Pruned $total_pruned weak connections")
end

"""
    visualize_token_network(module_name::String, max_tokens::Int=10)

Visualize token connections (for small networks).
"""
function visualize_token_network(module_name::String, max_tokens::Int=10)
    # Access global modules from TokenSystem
    modules = Main.TokenSystem.modules
    
    if !haskey(modules, module_name)
        println("Module $module_name not found")
        return
    end
    
    module_obj = modules[module_name]
    
    if isempty(module_obj.tokens)
        println("No tokens in module $module_name")
        return
    end
    
    # Get top tokens by importance
    top_tokens = sort(collect(module_obj.tokens), by=x->x[2].importance, rev=true)[1:min(max_tokens, length(module_obj.tokens))]
    
    println("\nToken Network for $module_name (Top $max_tokens tokens):")
    println("=============================================")
    
    for (i, (token_str, token_obj)) in enumerate(top_tokens)
        # Print token info
        println("$i. $token_str (Importance: $(round(token_obj.importance, digits=2)), Frequency: $(token_obj.frequency))")
        
        # Print connections
        if !isempty(token_obj.connections)
            connections = sort(collect(token_obj.connections), by=x->x[2], rev=true)[1:min(3, length(token_obj.connections))]
            println("   → $(join(["$(c[1])($(round(c[2], digits=2)))" for c in connections], ", "))")
        else
            println("   → No connections")
        end
    end
    
    # Print micro-models
    if !isempty(module_obj.micro_models)
        println("\nMicro-Models in $module_name:")
        for (i, model) in enumerate(module_obj.micro_models[max(1, length(module_obj.micro_models)-5):end])
            println("$i. $(join(model, " → "))")
        end
    end
    
    println("=============================================")
end

"""
    prune_low_importance_tokens(importance_threshold::Float64=0.3)

Remove tokens with low importance to reduce noise.
"""
function prune_low_importance_tokens(importance_threshold::Float64=0.3)
    # Access global modules from TokenSystem
    modules = Main.TokenSystem.modules
    
    total_pruned = 0
    
    for (name, module_obj) in modules
        # Find low-importance tokens
        low_importance_tokens = filter(p -> p.second.importance < importance_threshold, module_obj.tokens)
        
        # Remove low-importance tokens
        for (token_str, _) in low_importance_tokens
            delete!(module_obj.tokens, token_str)
            total_pruned += 1
        end
    end
    
    println("Pruned $total_pruned low-importance tokens")
end

end # module CompressionSystem
