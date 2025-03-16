module DataProcessing

using JSON3
using Random
using StatsBase

# Try to load CUDA if available
try
    using CUDA
    println("CUDA is available for GPU acceleration in data processing")
catch e
    println("CUDA not available in data processing, using CPU only: $e")
end

# Knowledge Queue structure for the delivery driver system
mutable struct KnowledgeQueue
    queue::Vector{Dict{String, Any}}  # Stores knowledge waiting for delivery
    capacity::Int                     # Max number of items in queue
    last_delivery_time::Float64       # Last time a delivery was made
end

# Global queues for each module
global knowledge_queues = Dict{String, KnowledgeQueue}()

"""
    initialize_knowledge_queues()

Initialize knowledge queues for each module in the token system.
"""
function initialize_knowledge_queues()
    # Import TokenSystem if not already in scope
    if !isdefined(Main, :TokenSystem)
        return
    end
    
    global knowledge_queues = Dict{String, KnowledgeQueue}()
    
    # Create a queue for each module
    for (name, _) in Main.TokenSystem.modules
        knowledge_queues[name] = KnowledgeQueue(
            Vector{Dict{String, Any}}(),  # Empty queue
            100,                          # Default capacity
            time()                        # Current time
        )
    end
    
    println("Initialized knowledge queues for $(length(knowledge_queues)) modules")
end

"""
    score_entry(entry)

Score an entry based on its complexity, novelty, and other factors.
Higher scores indicate more valuable knowledge.
"""
function score_entry(entry)
    # Base score
    score = 1.0
    
    # Prioritize complex or unique answers
    if haskey(entry, "answer") && length(entry["answer"]) > 50
        score += 0.5
    end
    
    # Extract domain
    domain = extract_domain_from_entry(entry)
    
    # Prioritize underrepresented domains
    if isdefined(Main, :TokenSystem) && haskey(Main.TokenSystem.modules, domain)
        if Main.TokenSystem.modules[domain].importance_score < 1.0
            score += 0.2  # Give a small boost
        end
    end
    
    # Prioritize entries with context
    if haskey(entry, "context") && !isempty(entry["context"])
        score += 0.3
    end
    
    # Add a small random factor to prevent stagnation
    score += 0.1 * rand()
    
    return score
end

"""
    enqueue_knowledge(text::String, metadata::Dict, module_name::String, priority::Float64)

Add knowledge to a module's queue for later delivery.
"""
function enqueue_knowledge(text::String, metadata::Dict, module_name::String, priority::Float64)
    if !haskey(knowledge_queues, module_name)
        return false
    end
    
    queue = knowledge_queues[module_name]
    
    # Check if queue is full
    if length(queue.queue) >= queue.capacity
        # If full, either drop the new entry or replace the lowest priority one
        if priority > 0.0 && !isempty(queue.queue)
            # Find the lowest priority entry
            min_priority_idx = argmin([get(entry, "priority", 0.0) for entry in queue.queue])
            min_priority = get(queue.queue[min_priority_idx], "priority", 0.0)
            
            # Replace if new entry has higher priority
            if priority > min_priority
                queue.queue[min_priority_idx] = Dict(
                    "text" => text,
                    "metadata" => metadata,
                    "priority" => priority,
                    "timestamp" => time()
                )
                return true
            else
                return false  # Couldn't add, queue full with higher priority items
            end
        else
            return false  # Queue full, couldn't add
        end
    else
        # Add to queue
        push!(queue.queue, Dict(
            "text" => text,
            "metadata" => metadata,
            "priority" => priority,
            "timestamp" => time()
        ))
        return true
    end
end

"""
    determine_best_delivery_module(tokens::Vector{String})

Determine the best module for delivering a set of tokens.
"""
function determine_best_delivery_module(tokens::Vector{String})
    if !isdefined(Main, :TokenSystem)
        return "Unknown"
    end
    
    allocations = Dict{String, Float64}()
    
    for (name, module_obj) in Main.TokenSystem.modules
        score = 0.0
        
        for token in tokens
            if haskey(module_obj.tokens, token)
                score += module_obj.tokens[token].importance  # Prioritize existing knowledge
            else
                score += 1.0  # Encourage new learning
            end
        end
        
        # Factor in module capacity and current queue size
        if haskey(knowledge_queues, name)
            queue_fullness = length(knowledge_queues[name].queue) / knowledge_queues[name].capacity
            score *= (1.0 - 0.5 * queue_fullness)  # Reduce score for modules with fuller queues
        end
        
        allocations[name] = score
    end
    
    # Return the module with the highest score
    if isempty(allocations)
        return "Unknown"
    else
        return first(sort(collect(allocations), by=x->x[2], rev=true))[1]
    end
end

"""
    deliver_knowledge(max_deliveries::Int=5)

Process the knowledge queues and deliver knowledge to modules.
Returns the number of deliveries made.
"""
function deliver_knowledge(max_deliveries::Int=5)
    if !isdefined(Main, :TokenSystem)
        return 0
    end
    
    deliveries_made = 0
    current_time = time()
    
    # Process each module's queue
    for (name, queue) in knowledge_queues
        # Skip empty queues
        if isempty(queue.queue)
            continue
        end
        
        # Enforce a minimum time between deliveries (rate limiting)
        if current_time - queue.last_delivery_time < 0.1  # 100ms minimum between deliveries
            continue
        end
        
        # Sort by priority (highest first)
        sort!(queue.queue, by=x->get(x, "priority", 0.0), rev=true)
        
        # Process up to max_deliveries entries
        deliveries_for_module = 0
        i = 1
        while i <= length(queue.queue) && deliveries_for_module < max_deliveries
            entry = queue.queue[i]
            
            # Process the knowledge
            text = entry["text"]
            metadata = entry["metadata"]
            
            # Use the TokenSystem to process the text
            try
                Main.TokenSystem.process_text(text, metadata)
                
                # Remove the entry from the queue
                deleteat!(queue.queue, i)
                
                # Update delivery stats
                deliveries_made += 1
                deliveries_for_module += 1
                queue.last_delivery_time = current_time
            catch e
                println("Error delivering knowledge to module $name: $e")
                i += 1  # Move to next entry
            end
        end
    end
    
    return deliveries_made
end

"""
    process_entry_with_delivery_driver(entry)

Process an entry using the delivery driver system instead of direct processing.
"""
function process_entry_with_delivery_driver(entry)
    # Extract text from the entry
    text = ""
    
    if haskey(entry, "question")
        text *= entry["question"] * " "
    end
    
    if haskey(entry, "answer")
        text *= entry["answer"] * " "
    end
    
    if haskey(entry, "context")
        text *= entry["context"] * " "
    end
    
    # Skip empty entries
    if isempty(text)
        return "Unknown"
    end
    
    # Tokenize the text
    tokens = split(lowercase(text))
    tokens = [String(w) for w in tokens]
    
    # Skip empty token lists
    if isempty(tokens)
        return "Unknown"
    end
    
    # Score the entry
    priority = score_entry(entry)
    
    # Determine the best module for delivery
    best_module = determine_best_delivery_module(tokens)
    
    # Create metadata
    metadata = Dict(
        "domain" => extract_domain_from_entry(entry),
        "concepts" => extract_concepts_from_entry(entry),
        "source" => get(entry, "source", "dataset"),
        "timestamp" => time()
    )
    
    # Enqueue the knowledge
    success = enqueue_knowledge(text, metadata, best_module, priority)
    
    if !success
        # If couldn't enqueue, try to process directly as fallback
        if isdefined(Main, :TokenSystem)
            try
                Main.TokenSystem.process_text(text, metadata)
            catch e
                println("Error in direct processing fallback: $e")
            end
        end
    end
    
    return best_module
end

"""
    load_dataset(file_path::String; max_entries::Int=0, sample_ratio::Float64=1.0)

Load a dataset from a JSONL file with optional sampling.
"""
function load_dataset(file_path::String; max_entries::Int=0, sample_ratio::Float64=1.0)
    if !isfile(file_path)
        println("Dataset file not found: $file_path")
        return []
    end
    
    println("Loading dataset from: $file_path")
    
    # Read all lines from the file
    lines = readlines(file_path)
    
    # Apply max_entries limit if specified
    if max_entries > 0 && max_entries < length(lines)
        println("Limiting to $max_entries entries (from $(length(lines)) total)")
        lines = lines[1:max_entries]
    end
    
    # Apply sampling if ratio is less than 1.0
    if sample_ratio < 1.0
        sample_size = max(1, round(Int, length(lines) * sample_ratio))
        println("Sampling $sample_size entries ($(round(sample_ratio * 100, digits=1))% of $(length(lines)))")
        lines = sample(lines, sample_size, replace=false)
    end
    
    # Parse each line as JSON
    dataset = []
    for line in lines
        try
            entry = JSON3.read(line)
            push!(dataset, entry)
        catch e
            println("Error parsing entry: $e")
        end
    end
    
    println("Loaded $(length(dataset)) entries")
    return dataset
end

"""
    prepare_batch(dataset, batch_size::Int)

Prepare batches from a dataset for processing.
"""
function prepare_batch(dataset, batch_size::Int)
    batches = []
    
    for i in 1:batch_size:length(dataset)
        end_idx = min(i + batch_size - 1, length(dataset))
        push!(batches, dataset[i:end_idx])
    end
    
    return batches
end

"""
    check_gpu_memory()

Check GPU memory usage if CUDA is available.
"""
function check_gpu_memory()
    if isdefined(Main, :CUDA) && Main.CUDA.functional()
        try
            # Get GPU memory info
            free_mem, total_mem = CUDA.memory_info()
            free_gb = free_mem / 1024^3
            total_gb = total_mem / 1024^3
            used_gb = total_gb - free_gb
            
            println("GPU memory: $(round(used_gb, digits=2))GB used / $(round(total_gb, digits=2))GB total ($(round(free_gb, digits=2))GB free)")
            
            # Warn if memory is low
            if free_gb < 1.0
                println("WARNING: Low GPU memory, consider reducing batch size or using CPU")
            end
        catch e
            println("Error checking GPU memory: $e")
        end
    end
end

"""
    tokenize_entry(entry)

Tokenize an entry for processing.
"""
function tokenize_entry(entry)
    # Extract text from the entry
    text = ""
    
    if haskey(entry, "question")
        text *= entry["question"] * " "
    end
    
    if haskey(entry, "answer")
        text *= entry["answer"] * " "
    end
    
    if haskey(entry, "context")
        text *= entry["context"] * " "
    end
    
    # Simple whitespace tokenization
    tokens = split(lowercase(text))
    
    # Filter out punctuation and short tokens
    tokens = filter(token -> length(token) > 1 && !all(c -> ispunct(c), token), tokens)
    
    return tokens
end

"""
    flag_entries_for_validation(dataset; ratio::Float64=0.05)

Flag a subset of entries for manual validation.
"""
# function flag_entries_for_validation(dataset; ratio::Float64=0.05)
#     # Determine how many entries to flag
#     num_to_flag = max(1, round(Int, length(dataset) * ratio))
#     
#     # Randomly select entries to flag
#     indices = sample(1:length(dataset), num_to_flag, replace=false)
#     
#     # Extract the flagged entries
#     flagged_entries = dataset[indices]
#     
#     println("Flagged $(length(flagged_entries)) entries for manual validation")
#     
#     return flagged_entries
# end

"""
    export_for_manual_review(entries, output_file::String)

Export entries for manual review.
"""
# function export_for_manual_review(entries, output_file::String)
#     open(output_file, "w") do io
#         for entry in entries
#             # Create a new dictionary with string keys
#             entry_with_validation = Dict{String, Any}()
#             
#             # Copy all existing key-value pairs
#             for (k, v) in pairs(entry)
#                 entry_with_validation[string(k)] = v
#             end
#             
#             # Add a validation field to the entry
#             entry_with_validation["validation"] = Dict{String, Any}(
#                 "is_valid" => nothing,
#                 "confidence" => nothing,
#                 "notes" => ""
#             )
#             
#             # Write the entry to the file
#             println(io, JSON3.write(entry_with_validation))
#         end
#     end
#     
#     println("Exported $(length(entries)) entries for manual review to $output_file")
# end

"""
    import_manual_review_results(input_file::String)

Import manual review results.
"""
# function import_manual_review_results(input_file::String)
#     if !isfile(input_file)
#         println("Manual review results file not found: $input_file")
#         return []
#     end
#     
#     # Read all lines from the file
#     lines = readlines(input_file)
#     
#     # Parse each line as JSON
#     results = []
#     for line in lines
#         try
#             entry = JSON3.read(line)
#             if haskey(entry, "validation")
#                 push!(results, entry)
#             end
#         catch e
#             println("Error parsing entry: $e")
#         end
#     end
#     
#     println("Imported $(length(results)) manual review results from $input_file")
#     
#     return results
# end

"""
    count_dataset_entries(file_path::String)

Count the total number of entries in a JSONL dataset file.
"""
function count_dataset_entries(file_path::String)
    if !isfile(file_path)
        println("Dataset file not found: $file_path")
        return 0
    end
    
    # Count lines in the file
    line_count = countlines(file_path)
    
    return line_count
end

"""
    load_dataset_chunk(file_path::String, start_idx::Int, end_idx::Int, sample_ratio::Float64=1.0)

Load a specific chunk of entries from a JSONL dataset file with optional sampling.
"""
function load_dataset_chunk(file_path::String, start_idx::Int, end_idx::Int, sample_ratio::Float64=1.0)
    if !isfile(file_path)
        println("Dataset file not found: $file_path")
        return []
    end
    
    # Read specific lines from the file
    open(file_path, "r") do file
        # Skip to start_idx
        for _ in 1:(start_idx-1)
            readline(file)
        end
        
        # Read lines from start_idx to end_idx
        lines = String[]
        for i in start_idx:end_idx
            line = readline(file)
            if eof(file) && isempty(line)
                break
            end
            push!(lines, line)
        end
        
        # Apply sampling if ratio is less than 1.0
        if sample_ratio < 1.0
            sample_size = max(1, round(Int, length(lines) * sample_ratio))
            println("Sampling $sample_size entries ($(round(sample_ratio * 100, digits=1))% of $(length(lines)))")
            lines = sample(lines, sample_size, replace=false)
        end
        
        # Parse each line as JSON
        dataset = []
        for line in lines
            try
                entry = JSON3.read(line)
                push!(dataset, entry)
            catch e
                println("Error parsing entry: $e")
            end
        end
        
        return dataset
    end
end

"""
    extract_domain_from_entry(entry)

Extract the domain from an entry based on its content.
"""
function extract_domain_from_entry(entry)
    # Default domain
    domain = "General"
    
    # Check if the entry has a domain field
    if haskey(entry, "domain")
        domain = entry["domain"]
    else
        # Try to infer the domain from the content
        text = ""
        
        if haskey(entry, "question")
            text *= entry["question"] * " "
        end
        
        if haskey(entry, "answer")
            text *= entry["answer"] * " "
        end
        
        if haskey(entry, "context")
            text *= entry["context"] * " "
        end
        
        # Simple keyword-based domain inference
        text = lowercase(text)
        
        if contains(text, "math") || contains(text, "equation") || contains(text, "calculus") || contains(text, "algebra")
            domain = "Math"
        elseif contains(text, "physics") || contains(text, "force") || contains(text, "energy") || contains(text, "motion")
            domain = "Physics"
        elseif contains(text, "language") || contains(text, "grammar") || contains(text, "syntax") || contains(text, "word")
            domain = "Language"
        elseif contains(text, "philosophy") || contains(text, "ethics") || contains(text, "moral") || contains(text, "logic")
            domain = "Philosophy"
        elseif contains(text, "memory") || contains(text, "recall") || contains(text, "remember") || contains(text, "forget")
            domain = "Memory"
        end
    end
    
    return domain
end

"""
    extract_concepts_from_entry(entry)

Extract potential concepts from an entry for verification.
"""
function extract_concepts_from_entry(entry)
    concepts = []
    
    # Extract text from the entry
    text = ""
    
    if haskey(entry, "question")
        text *= entry["question"] * " "
    end
    
    if haskey(entry, "answer")
        text *= entry["answer"] * " "
    end
    
    if haskey(entry, "context")
        text *= entry["context"] * " "
    end
    
    # Split into sentences (simple approach)
    sentences = split(text, r"[.!?]")
    
    # Extract noun phrases (simplified approach)
    for sentence in sentences
        # Skip short sentences
        if length(sentence) < 5
            continue
        end
        
        # Clean the sentence
        sentence = strip(sentence)
        
        # Add the whole sentence as a potential concept
        if length(sentence) > 0 && length(sentence) < 100
            push!(concepts, sentence)
        end
        
        # Extract phrases between commas
        phrases = split(sentence, ",")
        for phrase in phrases
            phrase = strip(phrase)
            if length(phrase) > 5 && length(phrase) < 50
                push!(concepts, phrase)
            end
        end
    end
    
    # Deduplicate concepts
    unique_concepts = unique(concepts)
    
    # Limit the number of concepts
    if length(unique_concepts) > 5
        unique_concepts = unique_concepts[1:5]
    end
    
    return unique_concepts
end

"""
    get_dataset_statistics(dataset)

Calculate statistics about a dataset.
"""
function get_dataset_statistics(dataset)
    # Calculate basic statistics
    num_entries = length(dataset)
    num_questions = sum(haskey(entry, "question") for entry in dataset)
    num_answers = sum(haskey(entry, "answer") for entry in dataset)
    num_contexts = sum(haskey(entry, "context") for entry in dataset)
    
    println("Dataset statistics:")
    println("  Number of entries: $num_entries")
    println("  Number of questions: $num_questions")
    println("  Number of answers: $num_answers")
    println("  Number of contexts: $num_contexts")
    
    return Dict(
        "num_entries" => num_entries,
        "num_questions" => num_questions,
        "num_answers" => num_answers,
        "num_contexts" => num_contexts
    )
end

"""
    train_in_waves(dataset_path::String; 
                   batch_size::Int=100, 
                   max_entries::Int=0,
                   model_path::String="trained_model.json")

Train the model in multiple waves, with each wave focusing on different aspects of learning.
Wave 1: Initial learning - broad coverage of the dataset
Wave 2: Refinement - focus on areas with weak knowledge
Wave 3: Refinement - focus on weak knowledge areas
"""
function train_in_waves(dataset_path::String; 
                        batch_size::Int=100, 
                        max_entries::Int=0,
                        model_path::String="trained_model.json")
    println("Starting wave-based training...")
    
    # Wave 1: Initial Learning
    println("\n=== Wave 1: Initial Learning ===")
    println("Focus: Broad coverage of dataset")
    
    # Train on a sample of the dataset
    wave1_model_path = replace(model_path, ".json" => "_wave1.json")
    
    if !isfile(wave1_model_path)
        println("Training Wave 1...")
        
        # For Wave 1, we use a lower sample ratio to quickly build a foundation
        if isdefined(Main, :TrainingSystem)
            Main.TrainingSystem.train_model(
                dataset_path,
                batch_size=batch_size,
                max_entries=max_entries > 0 ? round(Int, max_entries * 0.4) : 0,  # Use 40% of max entries
                model_path=wave1_model_path,
                compression_interval=1000,
                use_knowledge_economy=true
            )
        end
    else
        println("Wave 1 model already exists at $wave1_model_path")
    end
    
    # Wave 2: Focused Learning
    println("\n=== Wave 2: Focused Learning ===")
    println("Focus: Strengthening weak knowledge areas")
    
    # Identify weak knowledge areas from Wave 1
    weak_areas = []
    for (module_name, module_obj) in Main.TokenSystem.modules
        # Calculate average token importance
        if !isempty(module_obj.tokens)
            avg_importance = mean([token.importance for (_, token) in module_obj.tokens])
            
            if avg_importance < 0.5
                push!(weak_areas, module_name)
            end
        end
    end
    
    wave2_model_path = replace(model_path, ".json" => "_wave2.json")
    
    if !isfile(wave2_model_path)
        println("Training Wave 2...")
        
        # For Wave 2, we focus on specific domains
        if isdefined(Main, :TrainingSystem)
            # Load a new batch of data, potentially filtered for weak domains
            Main.TrainingSystem.train_model(
                dataset_path,
                batch_size=batch_size,
                max_entries=max_entries > 0 ? round(Int, max_entries * 0.3) : 0,  # Use 30% of max entries
                model_path=wave2_model_path,
                compression_interval=1000,
                use_knowledge_economy=true
            )
        end
    else
        println("Wave 2 model already exists at $wave2_model_path")
    end
    
    # Wave 3: Verification and Refinement
    println("\n=== Wave 3: Refinement ===")
    println("Focus: Fine-tuning knowledge")
    
    # Identify weak knowledge areas from Wave 2
    weak_areas = []
    for (module_name, module_obj) in Main.TokenSystem.modules
        # Calculate average token importance
        if !isempty(module_obj.tokens)
            avg_importance = mean([token.importance for (_, token) in module_obj.tokens])
            
            if avg_importance < 0.5
                push!(weak_areas, module_name)
            end
        end
    end
    
    # Train on the filtered dataset with high verification
    if isdefined(Main, :TrainingSystem)
        Main.TrainingSystem.train_model(
            dataset_path,
            batch_size=batch_size,
            max_entries=max_entries > 0 ? round(Int, max_entries * 0.3) : 0,  # Use 30% of max entries
            model_path=model_path,
            compression_interval=1000,
            use_knowledge_economy=true
        )
    end
    
    # Apply thorough compression after Wave 3
    if isdefined(Main, :CompressionSystem)
        println("Applying thorough compression after Wave 3...")
        Main.CompressionSystem.compress_modules(5, thorough=true)  # Higher threshold and thorough mode
    else
        println("CompressionSystem not available, skipping final compression")
    end
    
    println("Wave-based training completed. Final model saved to $model_path")
end

"""
    identify_weak_knowledge_areas()

Identify areas of knowledge that need reinforcement based on token importance.
"""
function identify_weak_knowledge_areas()
    weak_areas = []
    for (module_name, module_obj) in Main.TokenSystem.modules
        # Calculate average token importance
        if !isempty(module_obj.tokens)
            avg_importance = mean([token.importance for (_, token) in module_obj.tokens])
            
            if avg_importance < 0.5
                push!(weak_areas, module_name)
            end
        end
    end
    
    return weak_areas
end

"""
    create_filtered_dataset(input_path::String, output_path::String, weak_areas::Vector{String})

Create a filtered dataset focusing on weak knowledge areas.
"""
function create_filtered_dataset(input_path::String, output_path::String, weak_areas::Vector{String})
    if isempty(weak_areas)
        # If no weak areas identified, just copy the original dataset
        cp(input_path, output_path, force=true)
        println("No weak areas identified, using original dataset")
        return
    end
    
    # Load the dataset
    dataset = load_dataset(input_path)
    
    # Filter entries related to weak areas
    filtered_entries = []
    weak_area_modules = weak_areas
    
    for entry in dataset
        # Extract domain
        domain = extract_domain_from_entry(entry)
        
        # Include if domain is in weak areas
        if domain in weak_area_modules
            push!(filtered_entries, entry)
            continue
        end
        
        # Also include entries with tokens that appear in weak areas
        text = ""
        if haskey(entry, "question")
            text *= entry["question"] * " "
        end
        if haskey(entry, "answer")
            text *= entry["answer"] * " "
        end
        
        tokens = split(lowercase(text))
        tokens = [String(w) for w in tokens]
        
        # Check if any token is in a weak module
        for token in tokens
            for module_name in weak_area_modules
                if isdefined(Main, :TokenSystem) && 
                   haskey(Main.TokenSystem.modules, module_name) &&
                   haskey(Main.TokenSystem.modules[module_name].tokens, token)
                    push!(filtered_entries, entry)
                    break
                end
            end
        end
    end
    
    # Ensure we have enough entries
    if length(filtered_entries) < 100
        # If too few entries, add more from the original dataset
        remaining_entries = setdiff(dataset, filtered_entries)
        additional_needed = min(100 - length(filtered_entries), length(remaining_entries))
        
        if additional_needed > 0
            append!(filtered_entries, sample(remaining_entries, additional_needed, replace=false))
        end
    end
    
    # Write filtered dataset to file
    open(output_path, "w") do file
        for entry in filtered_entries
            println(file, JSON3.write(entry))
        end
    end
    
    println("Created filtered dataset with $(length(filtered_entries)) entries focusing on weak knowledge areas")
end

"""
    competitive_knowledge_entry(entry, existing_knowledge)

Determine if a new knowledge entry should replace existing knowledge based on quality and relevance.
Returns true if the new entry should be added/replace existing knowledge.
"""
function competitive_knowledge_entry(entry, existing_knowledge)
    # If no existing knowledge, always accept new knowledge
    if isempty(existing_knowledge)
        return true
    end
    
    # Score the new entry
    new_score = score_entry(entry)
    
    # Find the most similar existing entry
    most_similar_entry = nothing
    highest_similarity = 0.0
    
    for existing_entry in existing_knowledge
        similarity = calculate_entry_similarity(entry, existing_entry)
        
        if similarity > highest_similarity
            highest_similarity = similarity
            most_similar_entry = existing_entry
        end
    end
    
    # If no similar entry found, accept the new one
    if most_similar_entry === nothing || highest_similarity < 0.5
        return true
    end
    
    # Score the most similar existing entry
    existing_score = score_entry(most_similar_entry)
    
    # Accept new entry if it scores higher than the existing one
    return new_score > existing_score
end

"""
    calculate_entry_similarity(entry1, entry2)

Calculate similarity between two entries based on their content.
Returns a value between 0 (completely different) and 1 (identical).
"""
function calculate_entry_similarity(entry1, entry2)
    # Extract text from entries
    text1 = ""
    text2 = ""
    
    for field in ["question", "answer", "context"]
        if haskey(entry1, field)
            text1 *= entry1[field] * " "
        end
        
        if haskey(entry2, field)
            text2 *= entry2[field] * " "
        end
    end
    
    # Tokenize
    tokens1 = Set(split(lowercase(text1)))
    tokens2 = Set(split(lowercase(text2)))
    
    # Calculate Jaccard similarity
    intersection = length(intersect(tokens1, tokens2))
    union_size = length(union(tokens1, tokens2))
    
    if union_size == 0
        return 0.0
    end
    
    return intersection / union_size
end

end # module
