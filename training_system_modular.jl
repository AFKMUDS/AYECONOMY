#!/usr/bin/env julia

# Natural Reasoning Training System - Modular Implementation
# This is the main entry point that coordinates all modules

using Statistics
using Random
using StatsBase
using JSON3
using DataFrames
using ArgParse
using Dates

global AYE_GUARDRAILS_ENABLED = false
global AYE_GUARDRAILS_REPORT = false
global AYE_CONN_MAX = 0.35
global AYE_WORMHOLE_MAX = 1.0
global AYE_GLUE_IMPORTANCE_MAX = 1.0e-6
global AYE_GLUE_TOKENS = Set([
    "what", "is", "are", "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with", "as",
    "how", "do", "does", "did", "who", "why", "when", "where",
    "?", "!", ".", ","
])

global AYE_RL_NUDGE_ENABLED = false
global AYE_RL_ALPHA = 0.01
global AYE_RL_BETA = 0.03
global AYE_RL_SCOPE = Set([
    "what", "is", "are", "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with", "as",
    "how", "do", "does", "did", "who", "why", "when", "where",
    "means", "defined", "define", "because", "therefore", "thus", "so", "concept"
])

global AYE_TRAIN_PROBES_ENABLED = false
global AYE_TRAIN_PROBES_REPORT = false
global AYE_TRAIN_PROBES = String[
    "what is gravity?",
    "what is geometry?",
    "define energy",
    "what is topology?"
]

# Global timestamp for model versioning
global global_timestamp = string(Dates.now())

if get(ENV, "AYE_USE_CUDA", "0") == "1"
    try
        using CUDA
        println("CUDA is available for GPU acceleration")
    catch e
        println("CUDA not available, using CPU only: $e")
    end
end

function _get_module_top_tokens_map(model, module_name::String)
    if !haskey(model, "modules") || !haskey(model["modules"], module_name)
        return Dict{String, Any}()
    end
    module_data = model["modules"][module_name]
    tokens = get(module_data, "top_tokens", [])
    m = Dict{String, Any}()
    for t in tokens
        v = get(t, "value", nothing)
        if v !== nothing
            m[String(v)] = t
        end
    end
    return m
end

function _is_stop_token(t::AbstractString)
    return t in ("what", "is", "are", "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with", "as", "how", "do", "does", "did", "who", "why", "when", "where", "?", "!", ".", ",")
end

function _pick_start_token(tokens::Vector{String}, token_map::Dict{String, Any})
    best = nothing
    best_score = -Inf
    for t in tokens
        if !haskey(token_map, t)
            continue
        end
        imp = Float64(get(token_map[t], "importance", 0.0))
        score = imp
        if _is_stop_token(t)
            score *= 0.05
        end
        if score > best_score
            best_score = score
            best = t
        end
    end
    if best !== nothing
        return best
    end

    best2 = nothing
    best_imp = -Inf
    for (v, obj) in token_map
        imp = get(obj, "importance", 0.0)
        if imp > best_imp
            best_imp = imp
            best2 = v
        end
    end
    return best2
end

function _token_artifact_penalty(t::AbstractString)
    if isempty(t)
        return 10.0
    end
    letters = 0
    digits = 0
    punct = 0
    for c in t
        if isletter(c)
            letters += 1
        elseif isnumeric(c)
            digits += 1
        else
            punct += 1
        end
    end
    len = max(length(t), 1)
    p_ratio = punct / len
    penalty = 0.0
    if p_ratio >= 0.6
        penalty += 4.0
    elseif p_ratio >= 0.4
        penalty += 2.0
    end
    if occursin("@", t)
        penalty += 2.0
    end
    if occursin("\\", t)
        penalty += 3.0
    end
    if occursin("{", t) || occursin("}", t) || occursin("[", t) || occursin("]", t)
        penalty += 1.5
    end
    if t == "\"" || t == "'"
        penalty += 1.0
    end
    return penalty
end

function _softmax_sample(cands::Vector{Tuple{String, Float64}}; top_k::Int=12, temperature::Float64=0.9)
    if isempty(cands)
        return nothing
    end
    sort!(cands, by=x->x[2], rev=true)
    k = min(top_k, length(cands))
    sliced = cands[1:k]
    scores = [x[2] for x in sliced]
    m = maximum(scores)
    denom = 0.0
    probs = Vector{Float64}(undef, k)
    temp = max(temperature, 1e-6)
    for i in 1:k
        z = exp((scores[i] - m) / temp)
        probs[i] = z
        denom += z
    end
    if denom <= 0
        return sliced[1][1]
    end
    r = rand() * denom
    acc = 0.0
    for i in 1:k
        acc += probs[i]
        if r <= acc
            return sliced[i][1]
        end
    end
    return sliced[end][1]
end

function _choose_next_token(current::String, token_map::Dict{String, Any}, used::Set{String}, query_set::Set{String}, goal_token)
    if !haskey(token_map, current)
        return nothing
    end
    obj = token_map[current]
    conns = get(obj, "connections", [])
    wormholes = get(obj, "wormholes", [])
    if conns === nothing || isempty(conns)
        if wormholes === nothing || isempty(wormholes)
            return nothing
        end
    end

    cands = Tuple{String, Float64}[]
    for c in conns
        dst = get(c, "token", nothing)
        if dst === nothing
            continue
        end
        dsts = String(dst)
        strength = Float64(get(c, "strength", 0.0))
        dst_imp = 0.0
        if haskey(token_map, dsts)
            dst_imp = Float64(get(token_map[dsts], "importance", 0.0))
        end

        score = strength + 0.08 * log(1.0 + max(dst_imp, 0.0))

        if dsts in query_set
            score += 6.0
        end
        if goal_token !== nothing && dsts == goal_token
            score += 18.0
        end

        if _is_stop_token(dsts)
            score -= 1.25
        end

        score -= _token_artifact_penalty(dsts)

        if dsts in used
            score -= 2.5
        end

        push!(cands, (dsts, score))
    end

    if wormholes !== nothing && !isempty(wormholes)
        for w in wormholes
            dst = get(w, "token", nothing)
            if dst === nothing
                continue
            end
            dsts = String(dst)
            strength = Float64(get(w, "strength", 0.0))
            dist = Int(get(w, "distance", 2))

            dst_imp = 0.0
            if haskey(token_map, dsts)
                dst_imp = Float64(get(token_map[dsts], "importance", 0.0))
            end

            score = 0.70 * strength + 0.04 * log(1.0 + max(dst_imp, 0.0))
            score -= 0.15 * max(dist - 1, 0)

            if dsts in query_set
                score += 5.0
            end
            if goal_token !== nothing && dsts == goal_token
                score += 16.0
            end

            if _is_stop_token(dsts)
                score -= 1.0
            end
            score -= _token_artifact_penalty(dsts)
            if dsts in used
                score -= 2.0
            end

            push!(cands, (dsts, score))
        end
    end

    if isempty(cands)
        return nothing
    end

    return _softmax_sample(cands; top_k=12, temperature=0.85)
end

function _choose_next_token_with_meta(current::String, token_map::Dict{String, Any}, used::Set{String}, query_set::Set{String}, goal_token)
    if !haskey(token_map, current)
        return (nothing, nothing)
    end
    obj = token_map[current]
    conns = get(obj, "connections", [])
    wormholes = get(obj, "wormholes", [])
    if conns === nothing || isempty(conns)
        if wormholes === nothing || isempty(wormholes)
            return (nothing, nothing)
        end
    end

    cands = Tuple{String, Float64, Symbol}[]
    for c in conns
        dst = get(c, "token", nothing)
        if dst === nothing
            continue
        end
        dsts = String(dst)
        strength = Float64(get(c, "strength", 0.0))
        dst_imp = 0.0
        if haskey(token_map, dsts)
            dst_imp = Float64(get(token_map[dsts], "importance", 0.0))
        end

        score = strength + 0.08 * log(1.0 + max(dst_imp, 0.0))

        if dsts in query_set
            score += 6.0
        end
        if goal_token !== nothing && dsts == goal_token
            score += 18.0
        end

        if _is_stop_token(dsts)
            score -= 1.25
        end

        score -= _token_artifact_penalty(dsts)

        if dsts in used
            score -= 2.5
        end

        push!(cands, (dsts, score, :connections))
    end

    if wormholes !== nothing && !isempty(wormholes)
        for w in wormholes
            dst = get(w, "token", nothing)
            if dst === nothing
                continue
            end
            dsts = String(dst)
            strength = Float64(get(w, "strength", 0.0))
            dist = Int(get(w, "distance", 2))

            dst_imp = 0.0
            if haskey(token_map, dsts)
                dst_imp = Float64(get(token_map[dsts], "importance", 0.0))
            end

            score = 0.70 * strength + 0.04 * log(1.0 + max(dst_imp, 0.0))
            score -= 0.15 * max(dist - 1, 0)

            if dsts in query_set
                score += 5.0
            end
            if goal_token !== nothing && dsts == goal_token
                score += 16.0
            end

            if _is_stop_token(dsts)
                score -= 1.0
            end
            score -= _token_artifact_penalty(dsts)
            if dsts in used
                score -= 2.0
            end

            push!(cands, (dsts, score, :wormholes))
        end
    end

    if isempty(cands)
        return (nothing, nothing)
    end

    scored = [(a[1], a[2]) for a in cands]
    chosen = _softmax_sample(scored; top_k=12, temperature=0.85)
    if chosen === nothing
        return (nothing, nothing)
    end

    kind = nothing
    for c in cands
        if c[1] == chosen
            kind = c[3]
            break
        end
    end
    return (chosen, kind)
end

function _rl_apply_nudges!(model, top_module::String, steps::Vector{Tuple{String, String, Symbol}}; stalled::Bool=false, alpha::Float64=AYE_RL_ALPHA, beta::Float64=AYE_RL_BETA)
    if isempty(steps)
        return 0
    end
    if !haskey(model, "modules") || !haskey(model["modules"], top_module)
        return 0
    end
    moddata = model["modules"][top_module]
    top_tokens = get(moddata, "top_tokens", [])
    if top_tokens === nothing || !(top_tokens isa Vector)
        return 0
    end

    idx = Dict{String, Any}()
    for t in top_tokens
        if t isa Dict
            v = get(t, "value", nothing)
            if v !== nothing
                idx[String(v)] = t
            end
        end
    end

    updates = 0
    for (src, dst, kind) in steps
        if !(src in AYE_RL_SCOPE)
            continue
        end
        if !haskey(idx, src)
            continue
        end
        sobj = idx[src]
        edges = get(sobj, String(kind), [])
        if edges === nothing || !(edges isa Vector)
            continue
        end

        for e in edges
            if !(e isa Dict)
                continue
            end
            if String(get(e, "token", "")) != dst
                continue
            end
            s = Float64(get(e, "strength", 0.0))
            s *= (1.0 + alpha)
            hi = (kind == :connections) ? AYE_CONN_MAX : AYE_WORMHOLE_MAX
            if s < 0.0
                s = 0.0
            end
            if s > hi
                s = hi
            end
            e["strength"] = s
            updates += 1
            break
        end
    end

    if stalled && !isempty(steps)
        (src, dst, kind) = steps[end]
        if haskey(idx, src)
            sobj = idx[src]
            edges = get(sobj, String(kind), [])
            if edges !== nothing && (edges isa Vector)
                for e in edges
                    if e isa Dict && String(get(e, "token", "")) == dst
                        s = Float64(get(e, "strength", 0.0))
                        s *= max(0.0, (1.0 - beta))
                        if s < 0.0
                            s = 0.0
                        end
                        e["strength"] = s
                        updates += 1
                        break
                    end
                end
            end
        end
    end
    return updates
end

function _tokens_to_text(seq::Vector{String})
    if isempty(seq)
        return ""
    end
    no_space_before = Set([".", ",", ";", ":", "?", "!", ")", "]", "}"])
    no_space_after = Set(["(", "[", "{"])

    out = IOBuffer()
    prev = nothing
    for t in seq
        if prev === nothing
            print(out, t)
        else
            if t in no_space_before
                print(out, t)
            elseif prev in no_space_after
                print(out, t)
            else
                print(out, " ")
                print(out, t)
            end
        end
        prev = t
    end
    return String(take!(out))
end

function run_inference_conversational(model, query::String; max_tokens::Int=24, rl_nudge::Bool=AYE_RL_NUDGE_ENABLED, rl_save_path=nothing, silent::Bool=false)
    tokens = TokenSystem.tokenize_text(query)
    if isempty(tokens)
        if !silent
            println("(empty)")
        end
        return
    end

    math_run = _extract_math_run(tokens)
    goal_token = nothing
    if math_run !== nothing
        try
            result = _eval_math_tokens(math_run)
            result_str = if isfinite(result) && abs(result - round(result)) < 1e-12
                string(Int(round(result)))
            else
                string(result)
            end
            push!(tokens, "[calc]")
            push!(tokens, "=")
            push!(tokens, result_str)
            goal_token = result_str
        catch
        end
    end

    query_set = Set(tokens)

    activations = Dict{String, Float64}()
    for (name, module_data) in model["modules"]
        activation = 0.0
        for token_data in get(module_data, "top_tokens", [])
            token_value = get(token_data, "value", "")
            if token_value in tokens
                activation += get(token_data, "importance", 1.0)
            end
        end
        activation *= get(module_data, "importance_score", 1.0)
        activations[String(name)] = activation
    end
    sorted_activations = sort(collect(activations), by=x->x[2], rev=true)
    top_module = (!isempty(sorted_activations) && sorted_activations[1][2] > 0) ? sorted_activations[1][1] : "Language"

    token_map = _get_module_top_tokens_map(model, top_module)
    start = _pick_start_token(tokens, token_map)
    if start === nothing
        if !silent
            println("(no tokens available)")
        end
        return
    end

    seq = String[]
    used = Set{String}()
    push!(seq, start)
    push!(used, start)

    steps = Tuple{String, String, Symbol}[]

    current = start
    stalled = false
    for _ in 1:(max_tokens - 1)
        if rl_nudge
            (nxt, kind) = _choose_next_token_with_meta(current, token_map, used, query_set, goal_token)
            if nxt === nothing
                stalled = true
                break
            end
            if kind !== nothing
                push!(steps, (current, nxt, kind))
            end
        else
            nxt = _choose_next_token(current, token_map, used, query_set, goal_token)
            if nxt === nothing
                break
            end
        end

        if nxt === nothing
            break
        end
        push!(seq, nxt)
        push!(used, nxt)
        current = nxt
    end

    if rl_nudge
        updates = _rl_apply_nudges!(model, top_module, steps; stalled=stalled)
        if !silent && updates > 0
            println("(rl) updates=$(updates) stalled=$(stalled) module=$(top_module)")
        end
        if rl_save_path !== nothing
            try
                open(String(rl_save_path), "w") do io
                    JSON3.write(io, model)
                end
            catch
            end
        end
    end

    if !silent
        println(_tokens_to_text(seq))
    end
end

function _load_train_probes_from_file(path::String)
    if isempty(path) || !isfile(path)
        return nothing
    end
    lines = readlines(path)
    probes = String[]
    for ln in lines
        s = strip(ln)
        if isempty(s)
            continue
        end
        if startswith(s, "#")
            continue
        end
        push!(probes, s)
    end
    return probes
end

function _apply_training_probes!(model)
    if !AYE_TRAIN_PROBES_ENABLED
        return 0
    end
    ran = 0
    for p in AYE_TRAIN_PROBES
        try
            run_inference_conversational(model, p; rl_nudge=true, rl_save_path=nothing, silent=true)
            ran += 1
        catch
        end
    end
    return ran
end

# Include all modules
include("modules/data_processing.jl")
include("modules/token_system.jl")
include("modules/compression_system.jl")
include("modules/reasoning_system.jl")
include("modules/verification_system.jl")
include("modules/model_assembly.jl")

# Import modules
using .DataProcessing
using .TokenSystem
using .CompressionSystem
using .ReasoningSystem
using .VerificationSystem
using .ModelAssembly

"""
    load_existing_model(model_path::String)

Load an existing model from a JSON file, or initialize a new model if the file doesn't exist.
"""
function load_existing_model(model_path::String)
    if isfile(model_path)
        println("Loading existing model from: $model_path")
        try
            model_json = JSON3.read(read(model_path, String), Dict{String, Any})
            return model_json
        catch e
            println("Error loading model: $e")
            println("Initializing new model instead")
            return nothing
        end
    else
        println("No existing model found. Initializing new model.")
        TokenSystem.initialize_modules()
        return nothing
    end
end

"""
    process_entry(entry)

Process a single entry, allocating it to the most relevant module.
"""
function process_entry(entry)
    question = lowercase(get(entry, "question", ""))
    if isempty(question)
        return "Unknown"
    end
    
    words = TokenSystem.tokenize_text(question)
    
    # Skip empty entries
    if isempty(words)
        return "Unknown"
    end

    # Allocate to the most relevant module
    best_module = TokenSystem.allocate_compute(words)
    
    # Process tokens in the winning module
    for word in words
        TokenSystem.process_token(word, best_module)
    end
    
    # Build connections between tokens (micro-model formation)
    TokenSystem.build_connections(words, best_module)
    
    # Update module importance based on activity
    TokenSystem.modules[best_module].importance_score *= 1.01  # Small increase for active modules
    
    # Trigger inter-module communication for knowledge sharing
    ReasoningSystem.initiate_cross_module_reasoning(words, best_module)
    
    return best_module
end

"""
    process_entry_with_knowledge_economy(entry)

Process a single entry using the knowledge economy approach with the delivery driver.
"""
function process_entry_with_knowledge_economy(entry)
    # Use the delivery driver from DataProcessing module
    return DataProcessing.process_entry_with_delivery_driver(entry)
end

"""
    process_batch(batch, use_knowledge_economy::Bool=false)

Process a batch of entries, allocating each to the most relevant module.
"""
function process_batch(batch, use_knowledge_economy::Bool=false)
    module_counts = Dict{String, Int}()
    
    # Process each entry in the batch
    for entry in batch
        # Use either the standard processing or knowledge economy approach
        module_name = use_knowledge_economy ? 
                      process_entry_with_knowledge_economy(entry) : 
                      process_entry(entry)
        
        # Update module counts
        module_counts[module_name] = get(module_counts, module_name, 0) + 1
    end
    
    # If using knowledge economy, deliver knowledge from queues
    if use_knowledge_economy
        deliveries = DataProcessing.deliver_knowledge(10)  # Process up to 10 entries per call
        if deliveries > 0
            println("Delivered $deliveries knowledge entries from queues")
        end
    end
    
    return module_counts
end

"""
    train_model(dataset_path::String; 
               batch_size::Int=100, 
               max_entries::Int=0, 
               offset::Int=0,
               chunk_size::Int=100, 
               continuous::Bool=false,
               model_path::String="trained_model.json",
               compression_interval::Int=1000,
               use_knowledge_economy::Bool=true)

Train the model on a dataset.
"""
function train_model(dataset_path::String; 
                    batch_size::Int=100, 
                    max_entries::Int=0, 
                    offset::Int=0,
                    chunk_size::Int=100, 
                    continuous::Bool=false,
                    model_path::String="trained_model.json",
                    compression_interval::Int=1000,
                    use_knowledge_economy::Bool=true)
    # Load existing model or initialize new one
    model = load_existing_model(model_path)
    if model !== nothing
        TokenSystem.initialize_modules_from_model(model)
    end
    
    # Initialize knowledge queues if using knowledge economy
    if use_knowledge_economy
        DataProcessing.initialize_knowledge_queues()
        println("Using knowledge economy approach with delivery driver")
    end
    
    # Count total entries in the dataset
    total_entries = DataProcessing.count_dataset_entries(dataset_path)
    println("Total entries in dataset: $total_entries")
    
    # Determine how many entries to process
    entries_to_process = max_entries > 0 ? min(max_entries, total_entries - offset) : total_entries - offset
    println("Will process $entries_to_process entries starting from offset $offset")
    
    # Initialize progress tracking
    progress = 0.0
    last_progress_print = 0.0
    start_time = time()
    entries_processed = 0
    
    # Process in chunks for better memory management
    if continuous
        # Process the entire dataset in chunks until completion
        current_offset = offset
        while current_offset < offset + entries_to_process
            chunk_end = min(current_offset + chunk_size - 1, offset + entries_to_process - 1)
            chunk_size_actual = chunk_end - current_offset + 1
            
            # Load a chunk of the dataset
            chunk = DataProcessing.load_dataset_chunk(dataset_path, current_offset + 1, chunk_end + 1)
            
            if isempty(chunk)
                println("No entries found in chunk ($current_offset:$chunk_end)")
                break
            end
            
            # Process the chunk in batches
            for i in 1:batch_size:length(chunk)
                batch_end = min(i + batch_size - 1, length(chunk))
                batch = chunk[i:batch_end]
                
                # Process the batch
                module_counts = process_batch(batch, use_knowledge_economy)
                
                # Update progress
                entries_processed += length(batch)
                progress = entries_processed / entries_to_process
                
                # Print progress every 5%
                if progress - last_progress_print >= 0.05
                    elapsed_time = time() - start_time
                    entries_per_second = entries_processed / elapsed_time
                    estimated_remaining = (entries_to_process - entries_processed) / entries_per_second
                    
                    println("Progress: $(round(progress * 100, digits=1))% ($(entries_processed)/$(entries_to_process))")
                    println("  Rate: $(round(entries_per_second, digits=1)) entries/second")
                    println("  Estimated time remaining: $(round(estimated_remaining / 60, digits=1)) minutes")
                    println("  Module distribution: $(module_counts)")
                    
                    # Check GPU memory if available
                    DataProcessing.check_gpu_memory()
                    
                    last_progress_print = progress
                    
                    # Save intermediate model
                    save_model_to_file(model_path, progress=progress, last_processed_chunk="$(current_offset)-$(chunk_end)", total_entries=entries_processed)
                end
            end
            
            # Periodically compress knowledge to save memory
            if current_offset % compression_interval == 0
                CompressionSystem.compress_modules()
            end
            
            # Update offset for next chunk
            current_offset = chunk_end + 1
        end
    else
        # Load the dataset
        dataset = DataProcessing.load_dataset(dataset_path, max_entries=entries_to_process, sample_ratio=1.0)
        
        if isempty(dataset)
            println("No entries found in dataset")
            return
        end
        
        # Process the dataset in batches
        batches = DataProcessing.prepare_batch(dataset, batch_size)
        
        for (batch_idx, batch) in enumerate(batches)
            # Process the batch
            module_counts = process_batch(batch, use_knowledge_economy)
            
            # Update progress
            entries_processed += length(batch)
            progress = entries_processed / entries_to_process
            
            # Print progress every 5%
            if progress - last_progress_print >= 0.05
                elapsed_time = time() - start_time
                entries_per_second = entries_processed / elapsed_time
                estimated_remaining = (entries_to_process - entries_processed) / entries_per_second
                
                println("Progress: $(round(progress * 100, digits=1))% ($(entries_processed)/$(entries_to_process))")
                println("  Rate: $(round(entries_per_second, digits=1)) entries/second")
                println("  Estimated time remaining: $(round(estimated_remaining / 60, digits=1)) minutes")
                println("  Module distribution: $(module_counts)")
                
                # Check GPU memory if available
                DataProcessing.check_gpu_memory()
                
                last_progress_print = progress
                
                # Save intermediate model
                save_model_to_file(model_path, progress=progress, last_processed_chunk="batch_$(batch_idx)", total_entries=entries_processed)
            end
            
            # Periodically compress knowledge to save memory
            if batch_idx % (compression_interval ÷ batch_size) == 0
                CompressionSystem.compress_modules()
            end
        end
    end
    
    # Process any remaining entries in the knowledge queues
    if use_knowledge_economy
        println("Processing remaining entries in knowledge queues...")
        remaining_deliveries = 0
        while true
            deliveries = DataProcessing.deliver_knowledge(20)  # Process more entries in final cleanup
            if deliveries == 0
                break
            end
            remaining_deliveries += deliveries
        end
        println("Delivered $remaining_deliveries remaining knowledge entries from queues")
    end
    
    # Final compression
    CompressionSystem.compress_modules(thorough=true)
    
    # Save the final model
    save_model_to_file(model_path, progress=1.0, total_entries=entries_processed)
    
    # Print final statistics
    elapsed_time = time() - start_time
    println("Training completed in $(round(elapsed_time / 60, digits=1)) minutes")
    println("Processed $entries_processed entries")
    println("Final model saved to $model_path")
    
    return model_path
end

"""
    run_inference(model, query::String)

Run inference on a trained model with a query.
"""
function _is_number_token(t::AbstractString)
    s = String(t)
    return occursin(r"^(?:\d+(?:\.\d+)?|\.\d+)$", s)
end

function _is_op_token(t::AbstractString)
    return t in ("+", "-", "*", "/", "^")
end

function _is_math_token(t::AbstractString)
    return _is_number_token(t) || _is_op_token(t) || t in ("(", ")")
end

function _extract_math_run(tokens::Vector{String})
    best = nothing
    best_len = 0
    i = 1
    while i <= length(tokens)
        if !_is_math_token(tokens[i])
            i += 1
            continue
        end

        j = i
        has_op = false
        while j <= length(tokens) && _is_math_token(tokens[j])
            if _is_op_token(tokens[j])
                has_op = true
            end
            j += 1
        end

        len = j - i
        if has_op && len > best_len
            best = tokens[i:(j - 1)]
            best_len = len
        end

        i = j
    end

    return best
end

function _math_token_ratio(tokens::Vector{String})
    if isempty(tokens)
        return 0.0
    end
    math_count = count(t -> _is_math_token(t), tokens)
    return math_count / length(tokens)
end

function _precedence(op::AbstractString)
    if op == "^"
        return 4
    elseif op == "*" || op == "/"
        return 3
    elseif op == "+" || op == "-"
        return 2
    end
    return 0
end

function _right_associative(op::AbstractString)
    return op == "^"
end

function _apply_op(op::AbstractString, a::Float64, b::Float64)
    if op == "+"
        return a + b
    elseif op == "-"
        return a - b
    elseif op == "*"
        return a * b
    elseif op == "/"
        return a / b
    elseif op == "^"
        return a ^ b
    end
    error("Unknown operator: $op")
end

function _eval_math_tokens(expr_tokens::Vector{String})
    output = Any[]
    ops = String[]

    function pop_op!()
        op = pop!(ops)
        b = pop!(output)
        a = pop!(output)
        push!(output, _apply_op(op, Float64(a), Float64(b)))
    end

    i = 1
    while i <= length(expr_tokens)
        t = expr_tokens[i]

        if _is_number_token(t)
            push!(output, parse(Float64, t))
        elseif t == "("
            push!(ops, t)
        elseif t == ")"
            while !isempty(ops) && ops[end] != "("
                pop_op!()
            end
            if isempty(ops) || ops[end] != "("
                error("Mismatched parentheses")
            end
            pop!(ops)
        elseif _is_op_token(t)
            if t == "-" && (i == 1 || (expr_tokens[i-1] in ("(", "+", "-", "*", "/", "^")))
                push!(output, 0.0)
            end

            while !isempty(ops)
                top = ops[end]
                if top == "("
                    break
                end
                ptop = _precedence(top)
                pt = _precedence(t)
                if ptop > pt || (ptop == pt && !_right_associative(t))
                    pop_op!()
                else
                    break
                end
            end
            push!(ops, t)
        else
            error("Invalid token in expression: $t")
        end

        i += 1
    end

    while !isempty(ops)
        if ops[end] == "("
            error("Mismatched parentheses")
        end
        pop_op!()
    end

    if length(output) != 1
        error("Invalid expression")
    end

    return Float64(output[1])
end

function run_inference(model, query::String)
    println("\n=== RUNNING INFERENCE ===")
    println("Query: $query")
    
    # Tokenize query
    tokens = TokenSystem.tokenize_text(query)

    math_run = _extract_math_run(tokens)
    if math_run !== nothing
        try
            result = _eval_math_tokens(math_run)
            is_math_only = _math_token_ratio(tokens) >= 0.80

            result_str = if isfinite(result) && abs(result - round(result)) < 1e-12
                string(Int(round(result)))
            else
                string(result)
            end

            if is_math_only
                println("\nCalculator:")
                println("  Expression: $(join(math_run, " "))")
                println("  Result: $(result_str)")
                return
            else
                push!(tokens, "=")
                push!(tokens, result_str)
            end
        catch e
            println("\nCalculator error (falling back to graph inference): $e")
        end
    end
    
    # Skip if no tokens
    if isempty(tokens)
        println("Empty query")
        return
    end
    
    # Calculate module activations
    activations = Dict{String, Float64}()
    
    for (name, module_data) in model["modules"]
        activation = 0.0
        
        # Check token presence in top tokens
        for token_data in get(module_data, "top_tokens", [])
            token_value = get(token_data, "value", "")
            if token_value in tokens
                # Token match
                activation += get(token_data, "importance", 1.0)
            end
        end
        
        # Scale by module importance
        activation *= get(module_data, "importance_score", 1.0)
        
        activations[name] = activation
    end
    
    # Sort modules by activation
    sorted_activations = sort(collect(activations), by=x->x[2], rev=true)
    
    println("\nModule activations:")
    for (name, activation) in sorted_activations
        if activation > 0
            println("  $name: $(round(activation, digits=2))")
        end
    end
    
    # Get top activated module
    if !isempty(sorted_activations) && sorted_activations[1][2] > 0
        top_module = sorted_activations[1][1]
        println("\nTop module: $top_module")
        
        # Get verified knowledge from this domain
        verified_knowledge = get(model["verification"]["knowledge_base"], top_module, [])
        
        if !isempty(verified_knowledge)
            println("\nRelevant verified knowledge:")
            
            # Find relevant knowledge
            relevant_knowledge = []
            for concept in verified_knowledge
                # Simple relevance check: any token overlap
                concept_tokens = TokenSystem.tokenize_text(concept)
                if any(token in tokens for token in concept_tokens)
                    push!(relevant_knowledge, concept)
                end
            end
            
            # Show top relevant knowledge
            if !isempty(relevant_knowledge)
                for (i, concept) in enumerate(relevant_knowledge[1:min(5, length(relevant_knowledge))])
                    println("  $i. $concept")
                end
            else
                println("  No directly relevant knowledge found")
            end
        else
            println("\nNo verified knowledge in this domain")
        end
        
        # Get top tokens from this module
        top_tokens = get(model["modules"][top_module], "top_tokens", [])
        
        if !isempty(top_tokens)
            println("\nRelevant tokens and connections:")
            
            # Find relevant tokens
            for token_data in top_tokens
                token_value = get(token_data, "value", "")
                if token_value in tokens
                    println("  $token_value (Importance: $(round(get(token_data, "importance", 0.0), digits=2)))")
                    
                    # Show connections
                    connections = get(token_data, "connections", [])
                    if !isempty(connections)
                        println("    Connected to:")
                        for connection in connections
                            connected_token = get(connection, "token", "")
                            strength = get(connection, "strength", 0.0)
                            println("      $connected_token ($(round(strength, digits=2)))")
                        end
                    end
                end
            end
        end
    else
        println("\nNo module was significantly activated by this query")
    end
end

"""
    check_training_progress(model_path::String="trained_model.json")

Check the training progress from a saved model file.
"""
function check_training_progress(model_path::String="trained_model.json")
    if !isfile(model_path)
        println("No model file found at: $model_path")
        return
    end
    
    try
        model_json = JSON3.read(read(model_path, String))
        
        # Print basic model stats
        println("\n=== Model Statistics ===")
        if haskey(model_json, "stats")
            println("Total tokens: $(model_json["stats"]["total_tokens"])")
            println("Total references: $(model_json["stats"]["total_references"])")
            if haskey(model_json["stats"], "total_micro_models")
                println("Total micro models: $(model_json["stats"]["total_micro_models"])")
            end
        end
        
        # Print training progress
        println("\n=== Training Progress ===")
        if haskey(model_json, "metadata")
            metadata = model_json["metadata"]
            if haskey(metadata, "progress_percentage")
                println("Progress: $(metadata["progress_percentage"])%")
            end
            if haskey(metadata, "last_processed_chunk")
                println("Last processed chunk: $(metadata["last_processed_chunk"])")
            end
            if haskey(metadata, "total_entries")
                println("Total entries to process: $(metadata["total_entries"])")
            end
        else
            println("No training progress metadata found in the model.")
        end
        
        # Print verification stats
        println("\n=== Verification Statistics ===")
        if haskey(model_json, "verification") && haskey(model_json["verification"], "knowledge_base")
            kb = model_json["verification"]["knowledge_base"]
            total_concepts = 0
            println("Knowledge by domain:")
            for (domain, concepts) in kb
                concept_count = length(concepts)
                total_concepts += concept_count
                println("  $domain: $concept_count concepts")
            end
            println("Total verified concepts: $total_concepts")
        end
    catch e
        println("Error reading model file: $e")
    end
end

"""
    estimate_processing_time(dataset_path::String; 
                            batch_size::Int=100, 
                            sample_size::Int=100)

Estimate the time required to process the entire dataset.
"""
function estimate_processing_time(dataset_path::String; 
                                 batch_size::Int=100, 
                                 sample_size::Int=100)
    println("Estimating processing time for dataset: $dataset_path")
    
    # Initialize modules
    TokenSystem.initialize_modules()
    
    # Count total entries in dataset
    total_entries = DataProcessing.count_dataset_entries(dataset_path)
    println("Total entries in dataset: $total_entries")
    
    # Load a sample of the dataset
    println("Loading $sample_size sample entries for time estimation...")
    sample_data = DataProcessing.load_dataset_chunk(dataset_path, 1, sample_size, 1.0)
    
    # Process the sample in batches
    num_batches = ceil(Int, length(sample_data) / batch_size)
    println("Processing $sample_size entries in $num_batches batches...")
    
    start_time = Dates.now()
    
    for i in 1:num_batches
        batch_start = (i-1) * batch_size + 1
        batch_end = min(i * batch_size, length(sample_data))
        current_batch = sample_data[batch_start:batch_end]
        
        # Process the batch
        process_batch(current_batch)
    end
    
    # Calculate elapsed time
    elapsed_time = Dates.now() - start_time
    elapsed_seconds = Dates.value(elapsed_time) / 1000
    
    # Calculate processing speed
    entries_per_second = sample_size / elapsed_seconds
    
    # Estimate total processing time
    total_seconds = total_entries / entries_per_second
    minutes = total_seconds / 60
    hours = minutes / 60
    days = hours / 24
    
    # Print results
    println("\n=== TIME ESTIMATION RESULTS ===")
    println("Sample processing time: $(round(elapsed_seconds, digits=2)) seconds")
    println("Processing speed: $(round(entries_per_second, digits=2)) entries/second")
    
    println("\nEstimated time to process all $total_entries entries:")
    println("  - $(round(total_seconds, digits=2)) seconds")
    println("  - $(round(minutes, digits=2)) minutes")
    println("  - $(round(hours, digits=2)) hours")
    println("  - $(round(days, digits=2)) days")
    
    return Dict(
        "entries_per_second" => entries_per_second,
        "total_seconds" => total_seconds,
        "total_minutes" => minutes,
        "total_hours" => hours,
        "total_days" => days
    )
end

"""
    parse_commandline()

Parse command line arguments.
"""
function parse_commandline()
    s = ArgParseSettings()
    
    @add_arg_table s begin
        "command"
            help = "Command to run: train, train_waves, check_progress, inference, evaluate"
            required = true
        "--dataset", "-d"
            help = "Path to dataset file"
            default = "data/dataset.jsonl"
        "--model-path", "-m"
            help = "Path to save/load model"
            default = "trained_model.json"
        "--batch-size", "-b"
            help = "Batch size for training"
            arg_type = Int
            default = 100
        "--max-entries", "-e"
            help = "Maximum number of entries to process (0 for all)"
            arg_type = Int
            default = 0
        "--offset", "-o"
            help = "Offset to start processing from"
            arg_type = Int
            default = 0
        "--chunk-size", "-c"
            help = "Chunk size for continuous processing"
            arg_type = Int
            default = 1000
        "--query", "-q"
            help = "Query for inference"
            default = ""
        "--visualize", "-v"
            help = "Visualize training progress"
            action = :store_true
        "--use-api"
            help = "Use API for verification"
            action = :store_true
        "--resume"
            help = "Resume training from checkpoint"
            action = :store_true
        "--continuous"
            help = "Process the dataset continuously in batches until completion"
            action = :store_true
        "--use-knowledge-economy"
            help = "Use knowledge economy approach with delivery driver"
            action = :store_true
        "--wave-training"
            help = "Use wave-based training approach (deprecated, use train_waves command instead)"
            action = :store_true

        "--guardrails"
            help = "Enable training-time guardrails when saving models (caps connections/wormholes and clamps glue-token importance)"
            action = :store_true
        "--guardrails-report"
            help = "Print a short guardrails report at each save checkpoint"
            action = :store_true
        "--conn-max"
            help = "Maximum connection strength when guardrails are enabled"
            arg_type = Float64
            default = 0.35
        "--wormhole-max"
            help = "Maximum wormhole strength when guardrails are enabled"
            arg_type = Float64
            default = 1.0
        "--glue-imp-max"
            help = "Maximum importance allowed for glue tokens (what/is/the/of/...) when guardrails are enabled"
            arg_type = Float64
            default = 1.0e-6

        "--train-probes"
            help = "Enable training-time probe nudges by providing a probes text file (one prompt per line)"
            default = ""
        "--train-probes-report"
            help = "Print a one-line note at each save when training-time probes are enabled"
            action = :store_true
        "--train-probes-alpha"
            help = "Alpha reward multiplier for training-time probe nudges"
            arg_type = Float64
            default = 0.01
        "--train-probes-beta"
            help = "Beta penalty multiplier for training-time probe nudges"
            arg_type = Float64
            default = 0.03
    end
    
    return parse_args(s)
end

"""
    main()

Main entry point for the training system.
"""
function main()
    args = parse_commandline()

    global AYE_GUARDRAILS_ENABLED = get(args, "guardrails", false)
    global AYE_GUARDRAILS_REPORT = get(args, "guardrails-report", false)
    global AYE_CONN_MAX = Float64(get(args, "conn-max", 0.35))
    global AYE_WORMHOLE_MAX = Float64(get(args, "wormhole-max", 1.0))
    global AYE_GLUE_IMPORTANCE_MAX = Float64(get(args, "glue-imp-max", 1.0e-6))

    probes_path = String(get(args, "train-probes", ""))
    probes = _load_train_probes_from_file(probes_path)
    global AYE_TRAIN_PROBES_ENABLED = probes !== nothing && !isempty(probes)
    global AYE_TRAIN_PROBES_REPORT = get(args, "train-probes-report", false)
    if AYE_TRAIN_PROBES_ENABLED
        global AYE_TRAIN_PROBES = probes
        global AYE_RL_ALPHA = Float64(get(args, "train-probes-alpha", 0.01))
        global AYE_RL_BETA = Float64(get(args, "train-probes-beta", 0.03))
    end
    
    command = args["command"]
    
    if command == "train"
        train_model(
            args["dataset"],
            batch_size=args["batch-size"],
            max_entries=args["max-entries"],
            offset=args["offset"],
            chunk_size=args["chunk-size"], 
            continuous=args["continuous"],
            model_path=args["model-path"],
            use_knowledge_economy=args["use-knowledge-economy"]
        )
    elseif command == "train_waves"
        DataProcessing.train_in_waves(
            args["dataset"],
            batch_size=args["batch-size"],
            max_entries=args["max-entries"],
            model_path=args["model-path"]
        )
    elseif command == "check_progress"
        check_training_progress(args["model-path"])
    elseif command == "inference"
        run_inference(args["query"], args["model-path"])
    elseif command == "evaluate"
        evaluate_model(args["dataset"], args["model-path"], max_entries=args["max-entries"])
    else
        println("Unknown command: $command")
    end
end

"""
    save_model_to_file(file_path::String; progress::Float64=0.0, last_processed_chunk::String="", total_entries::Int=0)

Save the current model state to a file.
"""
function save_model_to_file(file_path::String; progress::Float64=0.0, last_processed_chunk::String="", total_entries::Int=0)
    # Safety check - ensure progress is defined
    if !@isdefined progress
        progress = 1.0  # Default to 100% if not defined
    end
    
    # Assemble the model
    model = Dict{String, Any}()
    
    # Add modules
    model["modules"] = Dict{String, Any}()
    for (name, module_data) in TokenSystem.modules
        model["modules"][name] = Dict{String, Any}(
            "importance_score" => module_data.importance_score,
            "top_tokens" => []
        )
        
        # Add top tokens
        sorted_tokens = sort(collect(module_data.tokens), by=x->x[2].importance, rev=true)

        local exported = Dict{String, Any}()
        local max_top = 1000

        for (token, token_data) in sorted_tokens[1:min(max_top, length(sorted_tokens))]
            token_info = Dict{String, Any}(
                "value" => token,
                "importance" => token_data.importance,
                "connections" => [],
                "wormholes" => []
            )
            
            # Add connections
            for (connected_token, strength) in token_data.connections
                push!(token_info["connections"], Dict{String, Any}(
                    "token" => connected_token,
                    "strength" => strength
                ))
            end

            # Add wormholes (2-hop path compression over the full in-memory graph)
            worm_scores = Dict{String, Float64}()
            for (mid, s1) in token_data.connections
                if !haskey(module_data.tokens, mid)
                    continue
                end
                mid_obj = module_data.tokens[mid]
                for (dst, s2) in mid_obj.connections
                    if dst == token
                        continue
                    end
                    if haskey(token_data.connections, dst)
                        continue
                    end
                    worm_scores[dst] = get(worm_scores, dst, 0.0) + Float64(s1) * Float64(s2)
                end
            end
            if !isempty(worm_scores)
                top_ws = sort(collect(worm_scores), by=x->x[2], rev=true)[1:min(8, length(worm_scores))]
                for (dst, sc) in top_ws
                    push!(token_info["wormholes"], Dict{String, Any}(
                        "token" => dst,
                        "strength" => sc,
                        "distance" => 2
                    ))
                end
            end
            
            push!(model["modules"][name]["top_tokens"], token_info)
            exported[String(token)] = token_info
        end

        # Ensure glue tokens remain available for module-local conversational inference.
        # Guardrails often clamp glue-token importance very low, which can push them out of the top-N export at scale.
        for gt in AYE_GLUE_TOKENS
            gts = String(gt)
            if haskey(module_data.tokens, gts) && !haskey(exported, gts)
                gtok = module_data.tokens[gts]
                token_info = Dict{String, Any}(
                    "value" => gts,
                    "importance" => gtok.importance,
                    "connections" => [],
                    "wormholes" => []
                )

                for (connected_token, strength) in gtok.connections
                    push!(token_info["connections"], Dict{String, Any}(
                        "token" => connected_token,
                        "strength" => strength
                    ))
                end

                worm_scores = Dict{String, Float64}()
                for (mid, s1) in gtok.connections
                    if !haskey(module_data.tokens, mid)
                        continue
                    end
                    mid_obj = module_data.tokens[mid]
                    for (dst, s2) in mid_obj.connections
                        if dst == gts
                            continue
                        end
                        if haskey(gtok.connections, dst)
                            continue
                        end
                        worm_scores[dst] = get(worm_scores, dst, 0.0) + Float64(s1) * Float64(s2)
                    end
                end
                if !isempty(worm_scores)
                    top_ws = sort(collect(worm_scores), by=x->x[2], rev=true)[1:min(8, length(worm_scores))]
                    for (dst, sc) in top_ws
                        push!(token_info["wormholes"], Dict{String, Any}(
                            "token" => dst,
                            "strength" => sc,
                            "distance" => 2
                        ))
                    end
                end

                push!(model["modules"][name]["top_tokens"], token_info)
                exported[gts] = token_info
            end
        end
        
        # Add micro-models
        model["modules"][name]["micro_models"] = []
        for micro_model in module_data.micro_models
            push!(model["modules"][name]["micro_models"], Dict{String, Any}(
                "tokens" => micro_model.tokens,
                "importance" => micro_model.importance
            ))
        end
    end
    
    # Add reasoning pathways
    model["reasoning_pathways"] = Dict{String, Any}()
    
    # Save reasoning pathways from module objects
    for (source_name, source_module) in TokenSystem.modules
        model["reasoning_pathways"][source_name] = Dict{String, Any}()
        
        for (target_name, pathways) in source_module.reasoning_pathways
            model["reasoning_pathways"][source_name][target_name] = []
            
            for (transform_type, strength) in pathways
                push!(model["reasoning_pathways"][source_name][target_name], Dict{String, Any}(
                    "type" => transform_type,
                    "strength" => strength
                ))
            end
        end
    end
    
    # Add verification data
    model["verification"] = Dict{String, Any}()
    
    # Safely access verification data with fallbacks
    try
        model["verification"] = Dict{String, Any}(
            "total_verifications" => VerificationSystem.global_verifier.total_verifications,
            "valid_concepts" => VerificationSystem.global_verifier.valid_concepts,
            "invalid_concepts" => VerificationSystem.global_verifier.invalid_concepts,
            "verification_methods" => VerificationSystem.global_verifier.verification_methods,
            "verified_domains" => VerificationSystem.global_verifier.verified_domains,
            "knowledge_base" => VerificationSystem.global_verifier.knowledge_base,
            "verification_history" => VerificationSystem.global_verifier.verification_history
        )
    catch e
        println("Warning: Some verification data could not be saved: $e")
        # Provide fallback values for critical fields
        if !haskey(model["verification"], "total_verifications")
            model["verification"]["total_verifications"] = 0
        end
        if !haskey(model["verification"], "valid_concepts")
            model["verification"]["valid_concepts"] = 0
        end
        if !haskey(model["verification"], "invalid_concepts")
            model["verification"]["invalid_concepts"] = 0
        end
    end
    
    # Add stats
    total_tokens = 0
    total_micro_models = 0
    total_references = 0
    
    for (_, module_data) in TokenSystem.modules
        total_tokens += length(module_data.tokens)
        total_micro_models += length(module_data.micro_models)
        
        # Count references (connections)
        for (_, token_data) in module_data.tokens
            total_references += length(token_data.connections)
        end
    end
    
    model["stats"] = Dict{String, Any}(
        "total_tokens" => total_tokens,
        "total_micro_models" => total_micro_models,
        "total_references" => total_references
    )
    
    # Add metadata
    model["metadata"] = Dict{String, Any}(
        "timestamp" => string(Dates.now()),
        "progress_percentage" => progress,
        "last_processed_chunk" => last_processed_chunk,
        "total_entries" => total_entries,
        "global_timestamp" => global_timestamp
    )

    if AYE_TRAIN_PROBES_ENABLED
        ran = _apply_training_probes!(model)
        if AYE_TRAIN_PROBES_REPORT
            println("(train-probes) ran=$(ran) alpha=$(AYE_RL_ALPHA) beta=$(AYE_RL_BETA)")
        end
    end

    function _clamp01(x::Float64, hi::Float64)
        if isnan(x) || !isfinite(x)
            return 0.0
        end
        if x < 0.0
            return 0.0
        end
        if x > hi
            return hi
        end
        return x
    end

    function _apply_guardrails!(m)
        if !AYE_GUARDRAILS_ENABLED
            return
        end
        if !haskey(m, "modules")
            return
        end
        for (modname, moddata) in m["modules"]
            top_tokens = get(moddata, "top_tokens", [])
            if top_tokens === nothing || !(top_tokens isa Vector)
                continue
            end
            for t in top_tokens
                if !(t isa Dict)
                    continue
                end
                v = get(t, "value", "")
                vstr = String(v)
                if vstr in AYE_GLUE_TOKENS
                    t["importance"] = min(Float64(get(t, "importance", 0.0)), AYE_GLUE_IMPORTANCE_MAX)
                end

                conns = get(t, "connections", [])
                if conns !== nothing && conns isa Vector
                    for c in conns
                        if c isa Dict && haskey(c, "strength")
                            c["strength"] = _clamp01(Float64(get(c, "strength", 0.0)), AYE_CONN_MAX)
                        end
                    end
                end

                whs = get(t, "wormholes", [])
                if whs !== nothing && whs isa Vector
                    for w in whs
                        if w isa Dict && haskey(w, "strength")
                            w["strength"] = _clamp01(Float64(get(w, "strength", 0.0)), AYE_WORMHOLE_MAX)
                        end
                    end
                end
            end
        end
    end

    function _guardrail_report(m)
        if !(AYE_GUARDRAILS_ENABLED && AYE_GUARDRAILS_REPORT)
            return
        end
        if !haskey(m, "modules")
            return
        end
        println("\n=== GUARDRAILS REPORT ===")
        println("conn_max=$(AYE_CONN_MAX) wormhole_max=$(AYE_WORMHOLE_MAX) glue_imp_max=$(AYE_GLUE_IMPORTANCE_MAX)")
        for (modname, moddata) in m["modules"]
            top_tokens = get(moddata, "top_tokens", [])
            if top_tokens === nothing || !(top_tokens isa Vector) || isempty(top_tokens)
                continue
            end

            top_worm = ("", -1.0)
            top_tok = ("", -1.0)

            for t in top_tokens
                if !(t isa Dict)
                    continue
                end
                v = String(get(t, "value", ""))
                imp = Float64(get(t, "importance", 0.0))
                if imp > top_tok[2]
                    top_tok = (v, imp)
                end
                whs = get(t, "wormholes", [])
                if whs !== nothing && whs isa Vector
                    for w in whs
                        if w isa Dict
                            sc = Float64(get(w, "strength", 0.0))
                            dst = String(get(w, "token", ""))
                            if sc > top_worm[2]
                                top_worm = ("$(v)~>$(dst)", sc)
                            end
                        end
                    end
                end
            end

            println("[$(String(modname))] top_token=$(top_tok[1]) imp=$(round(top_tok[2], digits=4))  top_wormhole=$(top_worm[1]) sc=$(round(top_worm[2], digits=4))")
        end
        println("=== END GUARDRAILS REPORT ===\n")
    end

    _apply_guardrails!(model)
    _guardrail_report(model)

    function _sanitize_float64(x::Float64)
        max_json_float = 1.0e300
        if isnan(x)
            return 0.0
        end
        if !isfinite(x)
            return signbit(x) ? -max_json_float : max_json_float
        end
        if abs(x) > max_json_float
            return x < 0 ? -max_json_float : max_json_float
        end
        return x
    end

    function _sanitize_json!(obj)
        if obj isa Dict
            for (k, v) in obj
                obj[k] = _sanitize_json!(v)
            end
            return obj
        elseif obj isa Vector
            for i in eachindex(obj)
                obj[i] = _sanitize_json!(obj[i])
            end
            return obj
        elseif obj isa Float64
            return _sanitize_float64(obj)
        elseif obj isa Real
            return obj
        else
            return obj
        end
    end

    _sanitize_json!(model)

    tmp_path = file_path * ".tmp"
    open(tmp_path, "w") do io
        JSON3.write(io, model)
    end
    mv(tmp_path, file_path; force=true)
    
    println("Model saved to $file_path")
    return model
end

"""
    load_model_from_file(file_path::String)

Load a model from a file.
"""
function load_model_from_file(file_path::String)
    if !isfile(file_path)
        println("Model file not found: $file_path")
        return nothing
    end
    
    try
        # Read the model file
        model_json = JSON3.read(read(file_path, String))
        
        # Initialize modules
        TokenSystem.initialize_modules()
        
        # Load modules
        if haskey(model_json, :modules)
            for (name, module_data) in model_json[:modules]
                if haskey(TokenSystem.modules, String(name))
                    # Set importance score
                    TokenSystem.modules[String(name)].importance_score = get(module_data, :importance_score, 1.0)
                    
                    # Load tokens
                    for token_data in get(module_data, :top_tokens, [])
                        token_value = get(token_data, :value, "")
                        if !isempty(token_value)
                            # Create token if it doesn't exist
                            if !haskey(TokenSystem.modules[String(name)].tokens, token_value)
                                TokenSystem.modules[String(name)].tokens[token_value] = TokenSystem.Token(
                                    token_value,
                                    get(token_data, :importance, 1.0),
                                    Dict{String, Float64}()
                                )
                            end
                            
                            # Load connections
                            for connection in get(token_data, :connections, [])
                                connected_token = get(connection, :token, "")
                                strength = get(connection, :strength, 0.0)
                                
                                if !isempty(connected_token)
                                    TokenSystem.modules[String(name)].tokens[token_value].connections[connected_token] = strength
                                end
                            end
                        end
                    end
                    
                    # Load micro-models
                    for micro_model_data in get(module_data, :micro_models, [])
                        tokens = get(micro_model_data, :tokens, [])
                        importance = get(micro_model_data, :importance, 1.0)
                        
                        if !isempty(tokens)
                            push!(TokenSystem.modules[String(name)].micro_models, TokenSystem.MicroModel(
                                tokens,
                                importance
                            ))
                        end
                    end
                end
            end
        end
        
        # Load reasoning pathways
        if haskey(model_json, :reasoning_pathways)
            TokenSystem.initialize_reasoning_pathways()
            
            for (source, targets) in model_json[:reasoning_pathways]
                source_str = String(source)
                if !haskey(TokenSystem.modules, source_str)
                    continue
                end
                
                for (target, pathways) in targets
                    target_str = String(target)
                    if !haskey(TokenSystem.modules, target_str)
                        continue
                    end
                    
                    # Clear existing pathways for this target
                    TokenSystem.modules[source_str].reasoning_pathways[target_str] = []
                    
                    for pathway_data in pathways
                        transform_type = get(pathway_data, :type, "composition")
                        strength = get(pathway_data, :strength, 0.5)
                        
                        # Add the pathway to the source module
                        push!(TokenSystem.modules[source_str].reasoning_pathways[target_str], 
                              (String(transform_type), strength))
                    end
                end
            end
            
            println("Initialized reasoning pathways between modules")
        else
            # If no reasoning pathways in the model, initialize them from scratch
            TokenSystem.initialize_reasoning_pathways()
        end
        
        # Load verification data
        if haskey(model_json, :verification)
            verification_data = model_json[:verification]
            
            # Reset verification system
            VerificationSystem.init_verification_system()
            
            # Load verification stats
            VerificationSystem.global_verifier.total_verifications = get(verification_data, :total_verifications, 0)
            VerificationSystem.global_verifier.valid_concepts = get(verification_data, :valid_concepts, 0)
            VerificationSystem.global_verifier.invalid_concepts = get(verification_data, :invalid_concepts, 0)
            
            # Load verification methods
            if haskey(verification_data, :verification_methods)
                for (method, count) in verification_data[:verification_methods]
                    VerificationSystem.global_verifier.verification_methods[String(method)] = count
                end
            end
            
            # Load verified domains
            if haskey(verification_data, :verified_domains)
                for (domain, count) in verification_data[:verified_domains]
                    VerificationSystem.global_verifier.verified_domains[String(domain)] = count
                end
            end
            
            # Load knowledge base
            if haskey(verification_data, :knowledge_base)
                for (domain, concepts) in verification_data[:knowledge_base]
                    VerificationSystem.global_verifier.knowledge_base[String(domain)] = concepts
                end
            end
            
            # Load verification history
            if haskey(verification_data, :verification_history)
                VerificationSystem.global_verifier.verification_history = verification_data[:verification_history]
            end
        end
        
        # Load global timestamp
        if haskey(model_json, :metadata) && haskey(model_json[:metadata], :global_timestamp)
            global global_timestamp = model_json[:metadata][:global_timestamp]
        end
        
        return model_json
    catch e
        println("Error loading model: $e")
        return nothing
    end
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
