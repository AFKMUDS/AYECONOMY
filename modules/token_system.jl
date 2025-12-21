module TokenSystem

using Statistics
using Random
using StatsBase

const GLUE_TOKENS = Set([
    "what", "is", "are", "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with", "as",
    "how", "do", "does", "did", "who", "why", "when", "where",
    "?", "!", ".", ","
])

const GLUE_CONTENT_REINFORCE_FACTOR = 0.04

# Constants for connection and token management
const CONNECTION_THRESHOLD = 0.5  # Threshold for significant connections between tokens
const MIN_TOKEN_IMPORTANCE = 0.2  # Minimum importance threshold for tokens to be retained

const IMPORTANCE_COMPRESSION_THRESHOLDS = (
    1e3, 1e6, 1e9, 1e12, 1e15, 1e18,
    1e21, 1e24, 1e27, 1e30, 1e33, 1e36,
    1e39, 1e42
)

function compress_importance(x::Float64)
    if !isfinite(x)
        return prevfloat(Inf)
    end

    y = x
    for t in IMPORTANCE_COMPRESSION_THRESHOLDS
        if y > t
            y = t + (y - t) / 2
        end
    end

    return y
end

function compress_importance(x::Real)
    return compress_importance(Float64(x))
end

export Token, KnowledgeModule, process_token, build_connections, initialize_modules, initialize_modules_from_model, process_text, compress_knowledge, prune_low_importance_tokens, tokenize_text, CONNECTION_THRESHOLD, MIN_TOKEN_IMPORTANCE, update_reasoning_pathways, initialize_reasoning_pathways

# Global timestamp for tracking token access
if !@isdefined global_timestamp
    global global_timestamp = 0
end

# Token structure for knowledge representation
mutable struct Token
    value::String
    frequency::Int
    importance::Float64
    connections::Dict{String, Float64}
    last_accessed::Int  # Timestamp
end

# Micro-model structure
mutable struct MicroModel
    tokens::Vector{String}
    importance::Float64
end

function tokenize_text(text::String)
    s = lowercase(text)
    tokens = String[]

    i = firstindex(s)
    lasti = lastindex(s)

    while i <= lasti
        c = s[i]

        if isspace(c)
            i = nextind(s, i)
            continue
        end

        if isletter(c) || c == '_'
            j = i
            j = nextind(s, j)
            while j <= lasti
                cj = s[j]
                if isletter(cj) || isnumeric(cj) || cj == '_'
                    j = nextind(s, j)
                else
                    break
                end
            end
            push!(tokens, String(SubString(s, i, prevind(s, j))))
            i = j
            continue
        end

        if isnumeric(c) || c == '.'
            j = i
            seen_digit = isnumeric(c)
            seen_dot = c == '.'
            j = nextind(s, j)
            while j <= lasti
                cj = s[j]
                if isnumeric(cj)
                    seen_digit = true
                    j = nextind(s, j)
                elseif cj == '.' && !seen_dot
                    seen_dot = true
                    j = nextind(s, j)
                else
                    break
                end
            end

            if seen_digit
                push!(tokens, String(SubString(s, i, prevind(s, j))))
                i = j
                continue
            end
        end

        if i < lasti
            c2 = s[nextind(s, i)]
            op2 = ""
            if (c == '<' && c2 == '=') || (c == '>' && c2 == '=') || (c == '!' && c2 == '=') ||
               (c == '=' && c2 == '=') || (c == '-' && c2 == '>') || (c == ':' && c2 == '=')
                op2 = string(c, c2)
            end
            if !isempty(op2)
                push!(tokens, op2)
                i = nextind(s, nextind(s, i))
                continue
            end
        end

        if ispunct(c) || c in ['+', '-', '*', '/', '^', '=', '<', '>', '(', ')', '[', ']', '{', '}', ',', ';', ':', '?', '!']
            push!(tokens, string(c))
            i = nextind(s, i)
            continue
        end

        i = nextind(s, i)
    end

    filter!(t -> !isempty(t), tokens)
    return tokens
end

# Knowledge module structure
mutable struct KnowledgeModule
    name::String
    tokens::Dict{String, Token}
    references::Dict{String, String}
    importance_score::Float64
    micro_models::Vector{MicroModel}
    message_queue::Vector{Dict{String, Any}}
    transformation_weights::Dict{String, Float64}
    reasoning_pathways::Dict{String, Vector{Tuple{String, Float64}}}
end

# Global modules dictionary
global modules = Dict{String, KnowledgeModule}()

"""
    initialize_modules()

Initialize knowledge modules for different domains.
"""
function initialize_modules()
    global modules = Dict{String, KnowledgeModule}()
    
    # Define initial modules
    module_names = ["Language", "Math", "Physics", "Philosophy", "Creativity", "Memory"]
    
    for name in module_names
        modules[name] = KnowledgeModule(
            name,                       # name
            Dict{String, Token}(),      # tokens
            Dict{String, String}(),     # references
            1.0,                        # importance_score
            Vector{MicroModel}(),       # micro_models
            Vector{Dict{String, Any}}(),# message_queue
            Dict{String, Float64}(),    # transformation_weights
            Dict{String, Vector{Tuple{String, Float64}}}() # reasoning_pathways
        )
    end
    
    println("Initialized $(length(modules)) knowledge modules")
    return modules
end

"""
    process_token(token::AbstractString, module_name::String)

Process a token in a module, updating its frequency, importance, and timestamp.
"""
function process_token(token::AbstractString, module_name::String)
    # Ensure global_timestamp is accessed as a global variable
    global global_timestamp
    
    # Convert to String type
    token_str = String(token)
    
    # Skip empty tokens
    if isempty(token_str)
        return
    end
    
    # Get module
    if !haskey(modules, module_name)
        error("Module not found: $module_name")
    end
    
    module_obj = modules[module_name]
    
    # Update global timestamp
    global_timestamp += 1
    
    # Check if token exists
    if haskey(module_obj.tokens, token_str)
        # Update existing token
        module_obj.tokens[token_str].frequency += 1
        
        # Apply value-based token reinforcement with adaptive decay
        # Tokens that are used more frequently get a smaller boost to prevent over-reinforcement
        # This implements a supply & demand system where common tokens have diminishing returns
        frequency_factor = 1.0 / (1.0 + 0.01 * module_obj.tokens[token_str].frequency)
        module_obj.tokens[token_str].importance *= 1.01 + (0.005 * frequency_factor)
        module_obj.tokens[token_str].importance = compress_importance(module_obj.tokens[token_str].importance)
        
        # Apply a small decay chance based on token importance
        # Higher importance tokens have a smaller chance of decay
        # This prevents rigid repetition and allows for more organic learning
        if rand() < 0.05 / (1.0 + module_obj.tokens[token_str].importance)
            module_obj.tokens[token_str].importance *= 0.99
            module_obj.tokens[token_str].importance = compress_importance(module_obj.tokens[token_str].importance)
        end
        
        module_obj.tokens[token_str].last_accessed = global_timestamp
    else
        # Create new token
        module_obj.tokens[token_str] = Token(token_str, 1, compress_importance(1.0), Dict(), global_timestamp)
    end
end

"""
    build_connections(tokens::Vector{<:AbstractString}, module_name::String)

Build connections between tokens in a sequence to form micro-models.
"""
function build_connections(tokens::Vector{<:AbstractString}, module_name::String)
    # Skip if less than 2 tokens
    if length(tokens) < 2
        return
    end
    
    # Get module
    if !haskey(modules, module_name)
        error("Module not found: $module_name")
    end
    
    module_obj = modules[module_name]
    
    # Convert tokens to String type
    string_tokens = [String(token) for token in tokens]
    
    # Build connections between adjacent tokens
    for i in 1:(length(string_tokens)-1)
        current = string_tokens[i]
        next = string_tokens[i+1]
        
        # Skip if either token doesn't exist (shouldn't happen)
        if !haskey(module_obj.tokens, current) || !haskey(module_obj.tokens, next)
            continue
        end
        
        current_is_glue = current in GLUE_TOKENS
        next_is_glue = next in GLUE_TOKENS
        reinforce_factor = (current_is_glue ⊻ next_is_glue) ? GLUE_CONTENT_REINFORCE_FACTOR : 1.0

        # Update or create connection with dynamic strength adjustment
        if haskey(module_obj.tokens[current].connections, next)
            # Connections that are reinforced frequently get diminishing returns
            current_strength = module_obj.tokens[current].connections[next]
            strength_factor = 1.0 / (1.0 + current_strength)
            module_obj.tokens[current].connections[next] += reinforce_factor * (0.1 * strength_factor)
        else
            module_obj.tokens[current].connections[next] = reinforce_factor * 0.1
        end
    end
    
    # Drop low-weight connections to encourage new ones
    # This ensures new connections form instead of just reinforcing old ones
    for token_str in string_tokens
        if haskey(module_obj.tokens, token_str)
            token_obj = module_obj.tokens[token_str]
            
            # Find weak connections
            weak_connections = []
            for (conn, strength) in token_obj.connections
                if strength < 0.2 && rand() < 0.1  # 10% chance of dropping weak links
                    push!(weak_connections, conn)
                end
            end
            
            # Remove weak connections
            for conn in weak_connections
                delete!(token_obj.connections, conn)
            end
        end
    end
    
    # Add as a micro-model if long enough, with competitive selection
    if length(string_tokens) >= 3
        # Calculate importance of this potential micro-model
        importance = sum(module_obj.tokens[token].importance for token in string_tokens) / length(string_tokens)
        
        # Check if this is a new micro-model
        if !in(string_tokens, [micro_model.tokens for micro_model in module_obj.micro_models])
            # Add the new micro-model
            push!(module_obj.micro_models, MicroModel(string_tokens, importance))
        else
            # Update existing micro-model importance
            for i in 1:length(module_obj.micro_models)
                if module_obj.micro_models[i].tokens == string_tokens
                    # Gradually update importance with new observations
                    module_obj.micro_models[i].importance = 0.9 * module_obj.micro_models[i].importance + 0.1 * importance
                    break
                end
            end
        end
        
        # Implement "Profit & Bankruptcy" - Remove weak micro-models
        # This means only the strongest concepts survive, making the AI evolve intelligently
        if length(module_obj.micro_models) > 50  # Limit the number of micro-models
            # Sort by importance
            sort!(module_obj.micro_models, by=m -> m.importance, rev=true)
            
            # Keep only the top models and those with high importance
            module_obj.micro_models = filter(m -> m.importance > 0.5 || m in module_obj.micro_models[1:min(30, length(module_obj.micro_models))], module_obj.micro_models)
        end
    end
end

"""
    get_token_importance(token::AbstractString, module_name::String)

Get the importance score of a token in a module.
"""
function get_token_importance(token::AbstractString, module_name::String)
    token_str = String(token)
    if !haskey(modules, module_name) || !haskey(modules[module_name].tokens, token_str)
        return 0.0
    end
    
    return modules[module_name].tokens[token_str].importance
end

"""
    get_related_tokens(token::AbstractString, module_name::String, max_tokens::Int=5)

Get the most strongly related tokens to a given token.
"""
function get_related_tokens(token::AbstractString, module_name::String, max_tokens::Int=5)
    token_str = String(token)
    if !haskey(modules, module_name) || !haskey(modules[module_name].tokens, token_str)
        return []
    end
    
    token_obj = modules[module_name].tokens[token_str]
    
    if isempty(token_obj.connections)
        return []
    end
    
    # Sort connections by strength
    sorted_connections = sort(collect(token_obj.connections), by=x->x[2], rev=true)
    
    # Return top connections
    return sorted_connections[1:min(max_tokens, length(sorted_connections))]
end

"""
    allocate_compute(tokens::Vector{<:AbstractString})

Allocate tokens to the most relevant module based on token importance.
"""
function allocate_compute(tokens::Vector{<:AbstractString})
    if isempty(modules)
        initialize_modules()
    end

    allocations = Dict{String, Float64}()
    
    for (name, module_obj) in modules
        score = 0.0
        for token in tokens
            if haskey(module_obj.tokens, String(token))
                # Known concept gets weighted by importance
                score += 0.5 * module_obj.tokens[String(token)].importance
            else
                # New learning opportunity
                score += 1.0
            end
        end
        allocations[name] = score
    end
    
    # Select the module with highest bid
    if isempty(allocations)
        return "Unknown"
    end
    return first(sort(collect(allocations), by=x->x[2], rev=true))[1]
end

"""
    add_transformation_type(module_name::String, transform_type::String, initial_weight::Float64=1.0)

Add a new transformation type to a module with an initial weight.
"""
function add_transformation_type(module_name::String, transform_type::String, initial_weight::Float64=1.0)
    if !haskey(modules, module_name)
        error("Module not found: $module_name")
    end
    
    module_obj = modules[module_name]
    
    # Add or update transformation weight
    module_obj.transformation_weights[transform_type] = get(module_obj.transformation_weights, transform_type, 0.0) + initial_weight
end

"""
    update_transformation_weight(module_name::String, transform_type::String, success::Bool, magnitude::Float64=0.1)

Update the weight of a transformation type based on success or failure.
"""
function update_transformation_weight(module_name::String, transform_type::String, success::Bool, magnitude::Float64=0.1)
    if !haskey(modules, module_name)
        error("Module not found: $module_name")
    end
    
    module_obj = modules[module_name]
    
    if !haskey(module_obj.transformation_weights, transform_type)
        module_obj.transformation_weights[transform_type] = 1.0
    end
    
    # Update weight based on success or failure
    if success
        module_obj.transformation_weights[transform_type] *= (1.0 + magnitude)
    else
        module_obj.transformation_weights[transform_type] *= (1.0 - magnitude)
    end
end

"""
    add_message_to_queue(source_module::String, target_module::String, message_type::String, content::Any)

Add a message to a module's message queue for inter-module communication.
"""
function add_message_to_queue(source_module::String, target_module::String, message_type::String, content::Any)
    if !haskey(modules, target_module)
        error("Target module not found: $target_module")
    end
    
    module_obj = modules[target_module]
    
    # Add message to queue
    push!(module_obj.message_queue, Dict(
        "source" => source_module,
        "type" => message_type,
        "content" => content,
        "timestamp" => global_timestamp
    ))
end

"""
    process_messages(module_name::String)

Process messages in a module's message queue.
"""
function process_messages(module_name::String)
    if !haskey(modules, module_name)
        error("Module not found: $module_name")
    end
    
    module_obj = modules[module_name]
    
    # Skip if no messages
    if isempty(module_obj.message_queue)
        return
    end
    
    # Process each message
    for message in module_obj.message_queue
        message_type = get(message, "type", "")
        content = get(message, "content", nothing)
        source = get(message, "source", "")
        
        if message_type == "token_request" && content isa String
            # Request for token information
            if haskey(module_obj.tokens, content)
                # Send token information back to source
                add_message_to_queue(
                    module_name,
                    source,
                    "token_response",
                    Dict(
                        "token" => content,
                        "importance" => module_obj.tokens[content].importance,
                        "connections" => module_obj.tokens[content].connections
                    )
                )
            end
        elseif message_type == "token_response" && content isa Dict
            # Response with token information
            token = get(content, "token", "")
            importance = get(content, "importance", 0.0)
            connections = get(content, "connections", Dict())
            
            # Use this information to enhance local knowledge
            if !isempty(token) && !haskey(module_obj.tokens, token)
                # Create new token with external information
                module_obj.tokens[token] = Token(token, 1, compress_importance(importance * 0.8), Dict(), global_timestamp)
                
                # Add some connections (with reduced strength)
                for (connected_token, strength) in connections
                    if haskey(module_obj.tokens, connected_token)
                        module_obj.tokens[token].connections[connected_token] = strength * 0.5
                    end
                end
            end
        elseif message_type == "transformation" && content isa Dict
            # Apply a transformation from another module
            transform_type = get(content, "type", "")
            source_tokens = get(content, "tokens", [])
            
            if !isempty(transform_type) && !isempty(source_tokens)
                # Record the transformation type
                add_transformation_type(module_name, transform_type)
                
                # Process the tokens
                for token in source_tokens
                    process_token(token, module_name)
                end
                
                # Build connections
                build_connections(source_tokens, module_name)
            end
        end
    end
    
    # Clear the message queue
    empty!(module_obj.message_queue)
end

"""
    initialize_reasoning_pathways()

Initialize reasoning pathways between modules.
"""
function initialize_reasoning_pathways()
    # Define initial transformation types
    transform_types = [
        "analogy",       # Finding similarities between concepts
        "abstraction",   # Moving from specific to general
        "concretization",# Moving from general to specific
        "causation",     # Identifying cause-effect relationships
        "negation",      # Reversing or contradicting a concept
        "composition",   # Combining multiple concepts
        "decomposition"  # Breaking down a concept
    ]
    
    # Initialize pathways for each module
    for (source_name, source_module) in modules
        for (target_name, _) in modules
            # Skip self-connections
            if source_name == target_name
                continue
            end
            
            # Initialize pathways with random transformation types
            source_module.reasoning_pathways[target_name] = []
            
            # Add 2-3 random transformation types
            num_transforms = rand(2:3)
            for _ in 1:num_transforms
                transform_type = rand(transform_types)
                initial_strength = 0.5 + 0.5 * rand()  # Random initial strength between 0.5 and 1.0
                
                push!(source_module.reasoning_pathways[target_name], (transform_type, initial_strength))
            end
        end
    end
    
    println("Initialized reasoning pathways between modules")
end

"""
    update_reasoning_pathways()

Update reasoning pathways based on successful transformations and module performance.
"""
function update_reasoning_pathways()
    for (name, module_obj) in modules
        # Skip if no transformation weights
        if isempty(module_obj.transformation_weights)
            continue
        end
        
        # Find most successful transformation types
        sorted_transforms = sort(collect(module_obj.transformation_weights), by=x->x[2], rev=true)
        
        # Update pathway strengths based on transformation success and module performance
        for (target, pathways) in module_obj.reasoning_pathways
            # Skip if target module doesn't exist
            if !haskey(modules, target)
                continue
            end
            
            # Get target module importance
            target_importance = modules[target].importance_score
            
            for i in 1:length(pathways)
                path_type, strength = pathways[i]
                
                # Find this transformation type in sorted list
                transform_idx = findfirst(x -> x[1] == path_type, sorted_transforms)
                
                if transform_idx !== nothing
                    # Adjust strength based on transformation success and target module importance
                    # Higher ranked transformations and more important target modules get stronger pathways
                    rank_factor = 1.0 - (transform_idx - 1) / length(sorted_transforms)
                    importance_factor = target_importance / 10.0  # Normalize importance
                    
                    # Apply market-driven adaptation - pathways to valuable modules get stronger
                    # Pathways using successful transformations also get stronger
                    # Add a small random factor to prevent stagnation (0.98-1.02 range)
                    adaptation_factor = 0.98 + 0.04 * rand()
                    new_strength = strength * (0.85 + 0.1 * rank_factor + 0.05 * importance_factor) * adaptation_factor
                    
                    # Update pathway strength
                    module_obj.reasoning_pathways[target][i] = (path_type, new_strength)
                end
            end
            
            # Sort pathways by strength (trade routes evolve based on which modules are doing well)
            if module_obj.importance_score > 1.1
                module_obj.reasoning_pathways[target] = sort!(pathways, by=x->x[2], rev=true)
                
                # Prune weak pathways and add new ones occasionally
                if !isempty(pathways) && length(pathways) > 1
                    # Remove weakest pathway if it's below threshold
                    if pathways[end][2] < 0.3 && rand() < 0.2
                        pop!(module_obj.reasoning_pathways[target])
                        
                        # Add a new pathway with a random transformation
                        if rand() < 0.5
                            transform_types = ["analogy", "abstraction", "concretization", "causation", 
                                              "negation", "composition", "decomposition"]
                            new_type = rand(transform_types)
                            push!(module_obj.reasoning_pathways[target], (new_type, 0.5 + 0.3 * rand()))
                        end
                    end
                end
            end
        end
    end
end

"""
    initiate_cross_module_reasoning(tokens::Vector{String}, source_module::String)

Initiate reasoning across modules based on a set of tokens.
"""
function initiate_cross_module_reasoning(tokens::Vector{String}, source_module::String)
    if !haskey(modules, source_module)
        error("Source module not found: $source_module")
    end
    
    source_obj = modules[source_module]
    
    # Skip if no tokens
    if isempty(tokens)
        return
    end
    
    # For each target module, check reasoning pathways
    for (target_module, pathways) in source_obj.reasoning_pathways
        # Skip if no pathways or target doesn't exist
        if isempty(pathways) || !haskey(modules, target_module)
            continue
        end
        
        # Choose a pathway based on strength (weighted random selection)
        weights = [strength for (_, strength) in pathways]
        idx = sample(1:length(pathways), Weights(weights))
        transform_type, _ = pathways[idx]
        
        # Apply the transformation
        transformed_tokens = apply_transformation(tokens, transform_type)
        
        # Send the transformed tokens to the target module
        add_message_to_queue(
            source_module,
            target_module,
            "transformation",
            Dict(
                "type" => transform_type,
                "tokens" => transformed_tokens
            )
        )
        
        # Update transformation weight based on success
        # In a real system, we would measure success more carefully
        update_transformation_weight(source_module, transform_type, true, 0.05)
    end
end

"""
    apply_transformation(tokens::Vector{String}, transform_type::String)

Apply a transformation to a set of tokens based on the transformation type.
Includes randomness to create varied responses.
"""
function apply_transformation(tokens::Vector{String}, transform_type::String)
    # Skip if no tokens
    if isempty(tokens)
        return []
    end
    
    # Add randomness to transformation selection (10% chance of using a different transformation)
    if rand() < 0.1
        transform_types = ["analogy", "abstraction", "concretization", "causation", 
                          "negation", "composition", "decomposition"]
        # Remove the current type from the list
        filter!(t -> t != transform_type, transform_types)
        # Select a random alternative transformation
        if !isempty(transform_types)
            transform_type = rand(transform_types)
        end
    end
    
    # Apply different transformations based on type
    if transform_type == "analogy"
        # For analogy, we replace some tokens with similar ones
        result = copy(tokens)
        if length(tokens) > 2 && rand() < 0.7
            # Replace 1-2 tokens with similar ones based on connections
            num_replacements = rand(1:min(2, length(tokens)))
            for _ in 1:num_replacements
                idx = rand(1:length(result))
                token = result[idx]
                
                # Find a connected token from any module
                replacement = token
                for (_, module_obj) in modules
                    if haskey(module_obj.tokens, token) && !isempty(module_obj.tokens[token].connections)
                        # Get a random connected token
                        connections = collect(keys(module_obj.tokens[token].connections))
                        if !isempty(connections)
                            replacement = rand(connections)
                            break
                        end
                    end
                end
                
                if replacement != token
                    result[idx] = replacement
                end
            end
        end
        return result
        
    elseif transform_type == "abstraction"
        # For abstraction, we remove specific details to create a more general concept
        if length(tokens) > 3
            # Keep a random subset of tokens, prioritizing higher importance tokens
            num_to_keep = max(2, rand(1:length(tokens)-1))
            
            # Try to keep the most important tokens
            token_importances = []
            for token in tokens
                importance = 0.0
                for (_, module_obj) in modules
                    if haskey(module_obj.tokens, token)
                        importance = max(importance, module_obj.tokens[token].importance)
                    end
                end
                push!(token_importances, (token, importance))
            end
            
            # Sort by importance and keep top tokens, plus add some randomness
            sort!(token_importances, by=x->x[2], rev=true)
            result_tokens = [t[1] for t in token_importances[1:min(num_to_keep, length(token_importances))]]
            
            # Add some randomness - 30% chance to include a random less important token
            if length(tokens) > num_to_keep && rand() < 0.3
                remaining = setdiff(tokens, result_tokens)
                if !isempty(remaining)
                    push!(result_tokens, rand(remaining))
                end
            end
            
            return result_tokens
        else
            return tokens
        end
        
    elseif transform_type == "concretization"
        # For concretization, we add specific details to make the concept more concrete
        result = copy(tokens)
        
        # 70% chance to add related tokens from micro-models
        if rand() < 0.7
            # Find micro-models containing these tokens
            related_tokens = Set{String}()
            for (_, module_obj) in modules
                for micro_model in module_obj.micro_models
                    # If any token in the input is in this micro-model
                    if any(token in tokens for token in micro_model.tokens)
                        # Add other tokens from the micro-model
                        for token in micro_model.tokens
                            if token ∉ tokens
                                push!(related_tokens, token)
                            end
                        end
                    end
                end
            end
            
            # Add 1-2 related tokens
            if !isempty(related_tokens)
                related_array = collect(related_tokens)
                num_to_add = rand(1:min(2, length(related_array)))
                for _ in 1:num_to_add
                    if !isempty(related_array)
                        token_to_add = rand(related_array)
                        push!(result, token_to_add)
                        filter!(t -> t != token_to_add, related_array)
                    end
                end
            end
        end
        
        return result
        
    elseif transform_type == "causation"
        # For causation, we reorder tokens to emphasize cause-effect relationships
        if length(tokens) > 2
            # Various causation patterns with randomness
            patterns = [
                # Move last token to front
                () -> [tokens[end], tokens[1:end-1]...],
                # Move first token to end
                () -> [tokens[2:end]..., tokens[1]],
                # Insert "because" or "therefore"
                () -> length(tokens) > 3 ? 
                      [tokens[1:div(length(tokens),2)]..., rand(["because", "therefore"]), tokens[div(length(tokens),2)+1:end]...] : 
                      tokens
            ]
            
            return rand(patterns)()
        else
            return tokens
        end
        
    elseif transform_type == "negation"
        # For negation, we add negation tokens or reverse meanings
        negation_options = [
            # Simple negation
            () -> ["not", tokens...],
            # Different negation words
            () -> [rand(["never", "cannot", "opposite_of"]), tokens...],
            # Negation in the middle
            () -> length(tokens) > 2 ? 
                  [tokens[1:div(length(tokens),2)]..., "not", tokens[div(length(tokens),2)+1:end]...] : 
                  ["not", tokens...]
        ]
        
        return rand(negation_options)()
        
    elseif transform_type == "composition"
        # For composition, we combine tokens from different sources
        result = copy(tokens)
        
        # Find tokens that frequently appear with these tokens
        composition_candidates = Set{String}()
        for token in tokens
            for (_, module_obj) in modules
                if haskey(module_obj.tokens, token)
                    # Add strongly connected tokens
                    for (connected, strength) in module_obj.tokens[token].connections
                        if strength > 0.3 && connected ∉ tokens
                            push!(composition_candidates, connected)
                        end
                    end
                end
            end
        end
        
        # Add some of these tokens
        if !isempty(composition_candidates)
            candidates_array = collect(composition_candidates)
            num_to_add = rand(1:min(3, length(candidates_array)))
            for _ in 1:num_to_add
                if !isempty(candidates_array)
                    token_to_add = rand(candidates_array)
                    push!(result, token_to_add)
                    filter!(t -> t != token_to_add, candidates_array)
                end
            end
        end
        
        return result
        
    elseif transform_type == "decomposition"
        # For decomposition, we break down concepts into components
        if length(tokens) > 1
            # Return a subset of tokens
            num_to_keep = max(1, rand(1:length(tokens)-1))
            indices = sample(1:length(tokens), num_to_keep, replace=false)
            return tokens[sort(indices)]
        else
            # Can't decompose a single token
            return tokens
        end
        
    else
        # Unknown transformation type - return with slight randomization
        if rand() < 0.3 && length(tokens) > 1
            # Shuffle some tokens
            result = copy(tokens)
            idx1, idx2 = sample(1:length(result), 2, replace=false)
            result[idx1], result[idx2] = result[idx2], result[idx1]
            return result
        else
            return tokens
        end
    end
end

"""
    initialize_modules_from_model(model)

Initialize knowledge modules from a previously trained model.
"""
function initialize_modules_from_model(model)
    global modules = Dict{String, KnowledgeModule}()
    
    # Check if model has modules
    if !haskey(model, "modules")
        println("Model does not contain modules. Initializing default modules instead.")
        return initialize_modules()
    end
    
    # Initialize modules from the model
    for (name, module_data) in model["modules"]
        # Convert name to String if it's a Symbol
        name_str = String(name)
        
        # Create a new module
        modules[name_str] = KnowledgeModule(
            name_str,                    # name
            Dict{String, Token}(),      # tokens
            Dict{String, String}(),     # references
            get(module_data, "importance_score", 1.0), # importance_score
            Vector{MicroModel}(),       # micro_models
            Vector{Dict{String, Any}}(),# message_queue
            Dict{String, Float64}(),    # transformation_weights
            Dict{String, Vector{Tuple{String, Float64}}}() # reasoning_pathways
        )
        
        # Restore tokens
        if haskey(module_data, "top_tokens")
            for token_data in module_data["top_tokens"]
                token_value = get(token_data, "value", "")
                if !isempty(token_value)
                    # Create token
                    token = Token(
                        token_value,
                        get(token_data, "frequency", 1),
                        compress_importance(get(token_data, "importance", 1.0)),
                        Dict{String, Float64}(),
                        0  # last_accessed will be updated on first access
                    )
                    
                    # Restore connections
                    if haskey(token_data, "connections")
                        for connection in token_data["connections"]
                            connected_token = get(connection, "token", "")
                            strength = get(connection, "strength", 0.1)
                            if !isempty(connected_token)
                                token.connections[connected_token] = strength
                            end
                        end
                    end
                    
                    # Add token to module
                    modules[name_str].tokens[token_value] = token
                end
            end
        end
        
        # Restore micro_models
        if haskey(module_data, "micro_models")
            for micro_model_data in module_data["micro_models"]
                if isa(micro_model_data, Vector)
                    importance = sum(modules[name_str].tokens[token].importance for token in micro_model_data) / length(micro_model_data)
                    push!(modules[name_str].micro_models, MicroModel(micro_model_data, importance))
                end
            end
        end
        
        # Restore reasoning pathways
        if haskey(module_data, "reasoning_pathways")
            for (target_module, pathways) in module_data["reasoning_pathways"]
                pathway_tuples = Vector{Tuple{String, Float64}}()
                for pathway in pathways
                    if isa(pathway, Vector) && length(pathway) == 2
                        push!(pathway_tuples, (pathway[1], pathway[2]))
                    end
                end
                modules[name_str].reasoning_pathways[target_module] = pathway_tuples
            end
        end
    end
    
    # Initialize reasoning pathways if not present
    initialize_reasoning_pathways()
    
    println("Initialized $(length(modules)) knowledge modules from existing model")
    return modules
end

"""
    process_text(text::String, metadata::Dict)

Process a text string with associated metadata, extracting tokens and allocating them to modules.
"""
function process_text(text::String, metadata::Dict)
    # Skip empty text
    if isempty(text)
        return
    end
    
    words = tokenize_text(text)
    
    # Skip empty entries
    if isempty(words)
        return
    end

    # Allocate to the most relevant module
    best_module = allocate_compute(words)
    
    # Process tokens in the winning module
    for word in words
        process_token(word, best_module)
    end
    
    # Build connections between tokens (micro-model formation)
    build_connections(words, best_module)
    
    # Update module importance based on activity
    modules[best_module].importance_score *= 1.01  # Small increase for active modules
    
    # Trigger inter-module communication for knowledge sharing
    initiate_cross_module_reasoning(words, best_module)
    
    # Periodically verify knowledge (not every entry to save computation)
    if rand() < 0.1  # 10% chance
        # Extract a meaningful phrase for verification
        if length(words) >= 5
            start_idx = rand(1:max(1, length(words)-4))
            phrase = join(words[start_idx:min(length(words), start_idx+4)], " ")
            
            # Add a message to the module's queue
            add_message_to_queue(best_module, best_module, "knowledge_update", phrase)
        end
    end
    
    return best_module
end

"""
    compress_knowledge(; thorough::Bool=false)

Compress the knowledge in all modules by merging similar tokens and pruning low-importance ones.
Returns statistics about the compression process.
"""
function compress_knowledge(; thorough::Bool=false)
    total_tokens_before = 0
    total_references_before = 0
    
    # Count tokens and references before compression
    for (_, module_data) in modules
        total_tokens_before += length(module_data.tokens)
        
        # Count references (connections)
        for (_, token_data) in module_data.tokens
            total_references_before += length(token_data.connections)
        end
    end
    
    # Threshold for merging tokens
    threshold = thorough ? 0.2 : 0.4  # Lower threshold for thorough compression
    
    # Compress each module
    for (module_name, module_data) in modules
        # Get all tokens that currently exist in this module
        existing_tokens = filter(token -> haskey(module_data.tokens, token), collect(keys(module_data.tokens)))
        
        # Skip if too few tokens
        if length(existing_tokens) < 2
            continue
        end
        
        # Cluster tokens based on connections
        clusters = Vector{Vector{String}}()
        
        for token in existing_tokens
            # Skip if token no longer exists (might have been removed during the process)
            if !haskey(module_data.tokens, token)
                continue
            end
            
            # Find most similar cluster
            max_similarity = 0.0
            cluster_idx = 0
            
            for (i, cluster) in enumerate(clusters)
                # Calculate average similarity with this cluster
                cluster_similarity = 0.0
                valid_comparisons = 0
                
                for cluster_token in cluster
                    # Skip if cluster token no longer exists
                    if !haskey(module_data.tokens, cluster_token)
                        continue
                    end
                    
                    # Calculate similarity
                    token_similarity = calculate_token_similarity(token, cluster_token, module_data.tokens)
                    
                    # Only count valid comparisons
                    if token_similarity > 0.0
                        cluster_similarity += token_similarity
                        valid_comparisons += 1
                    end
                end
                
                # Calculate average similarity
                avg_similarity = valid_comparisons > 0 ? cluster_similarity / valid_comparisons : 0.0
                
                # Update max similarity
                if avg_similarity > max_similarity
                    max_similarity = avg_similarity
                    cluster_idx = i
                end
            end
            
            # Add to cluster or create new one
            if max_similarity < 0.3 || cluster_idx == 0
                # Create a new cluster
                push!(clusters, [token])
            else
                # Add to existing cluster
                push!(clusters[cluster_idx], token)
            end
        end
        
        # Build micro-models from clusters
        for cluster in clusters
            # Filter out any tokens that might have been removed
            valid_tokens = filter(token -> haskey(module_data.tokens, token), cluster)
            
            # Skip if no valid tokens remain
            if isempty(valid_tokens)
                continue
            end
            
            # Sort tokens by importance
            sorted_cluster = sort(valid_tokens, by=x->module_data.tokens[x].importance, rev=true)
            
            # Add as a micro-model
            importance = sum(module_data.tokens[token].importance for token in sorted_cluster) / length(sorted_cluster)
            push!(module_data.micro_models, MicroModel(sorted_cluster, importance))
        end
    end
    
    # Count tokens and references after compression
    total_tokens_after = 0
    total_references_after = 0
    
    for (_, module_data) in modules
        total_tokens_after += length(module_data.tokens)
        
        # Count references (connections)
        for (_, token_data) in module_data.tokens
            total_references_after += length(token_data.connections)
        end
    end
    
    # Prune low-importance tokens if thorough
    pruned = 0
    if thorough
        pruned = prune_low_importance_tokens(0.2)  # More aggressive pruning for thorough compression
    end
    
    # Return compression stats
    return Dict(
        "tokens" => total_tokens_after,
        "references" => total_references_after,
        "ratio" => total_references_after / max(1, total_tokens_after),
        "pruned" => pruned
    )
end

"""
    prune_low_importance_tokens(threshold::Float64=0.1)

Prune tokens with importance below the threshold.
Returns the number of tokens pruned.
"""
function prune_low_importance_tokens(threshold::Float64=0.1)
    pruned_count = 0
    
    for (_, module_data) in modules
        # Get tokens to prune
        tokens_to_prune = []
        
        for (token, token_data) in module_data.tokens
            if token_data.importance < threshold
                push!(tokens_to_prune, token)
            end
        end
        
        # Prune tokens
        for token in tokens_to_prune
            delete!(module_data.tokens, token)
            pruned_count += 1
        end
        
        # Update micro-models
        i = 1
        while i <= length(module_data.micro_models)
            micro_model = module_data.micro_models[i]
            
            # Check if any tokens in the micro-model were pruned
            if any(token -> !haskey(module_data.tokens, token), micro_model.tokens)
                # Remove this micro-model
                deleteat!(module_data.micro_models, i)
            else
                i += 1
            end
        end
    end
    
    return pruned_count
end

"""
    calculate_token_similarity(token1::String, token2::String, tokens::Dict{String, Token})

Calculate the similarity between two tokens based on their connections.
"""
function calculate_token_similarity(token1::String, token2::String, tokens::Dict{String, Token})
    # Check if both tokens exist in the dictionary
    if !haskey(tokens, token1) || !haskey(tokens, token2)
        # Only print a warning for debugging purposes, not in production
        # if !haskey(tokens, token1)
        #     println("Skipping missing token: $(token1)")
        # end
        # if !haskey(tokens, token2)
        #     println("Skipping missing token: $(token2)")
        # end
        # Return default similarity
        return 0.0
    end

    # Get connections for both tokens
    connections1 = tokens[token1].connections
    connections2 = tokens[token2].connections
    
    # Get common connections
    common_connections = intersect(keys(connections1), keys(connections2))
    
    # Calculate similarity based on common connections
    similarity = 0.0
    
    if !isempty(common_connections)
        # Calculate similarity as the average of the connection strength differences
        total_diff = 0.0
        
        for conn in common_connections
            diff = abs(connections1[conn] - connections2[conn])
            total_diff += diff
        end
        
        # Normalize
        avg_diff = total_diff / length(common_connections)
        similarity = 1.0 - avg_diff
    end
    
    # Adjust similarity based on the number of common connections
    total_connections = length(union(keys(connections1), keys(connections2)))
    if total_connections > 0
        similarity *= length(common_connections) / total_connections
    end
    
    return similarity
end

"""
    build_micro_models(module_name::String)

Build micro-models from token clusters in the specified module.
"""
function build_micro_models(module_name::String)
    # Get module
    if !haskey(modules, module_name)
        return
    end
    
    module_obj = modules[module_name]
    
    # Get all tokens that currently exist in this module
    existing_tokens = filter(token -> haskey(module_obj.tokens, token), collect(keys(module_obj.tokens)))
    
    # Skip if too few tokens
    if length(existing_tokens) < 2
        return
    end
    
    # Cluster tokens based on connections
    clusters = Vector{Vector{String}}()
    
    for token in existing_tokens
        # Skip if token no longer exists (might have been removed during the process)
        if !haskey(module_obj.tokens, token)
            continue
        end
        
        # Find most similar cluster
        max_similarity = 0.0
        cluster_idx = 0
        
        for (i, cluster) in enumerate(clusters)
            # Calculate average similarity with this cluster
            cluster_similarity = 0.0
            valid_comparisons = 0
            
            for cluster_token in cluster
                # Skip if cluster token no longer exists
                if !haskey(module_obj.tokens, cluster_token)
                    continue
                end
                
                # Calculate similarity
                token_similarity = calculate_token_similarity(token, cluster_token, module_obj.tokens)
                
                # Only count valid comparisons
                if token_similarity > 0.0
                    cluster_similarity += token_similarity
                    valid_comparisons += 1
                end
            end
            
            # Calculate average similarity
            avg_similarity = valid_comparisons > 0 ? cluster_similarity / valid_comparisons : 0.0
            
            # Update max similarity
            if avg_similarity > max_similarity
                max_similarity = avg_similarity
                cluster_idx = i
            end
        end
        
        # Add to cluster or create new one
        if max_similarity < 0.3 || cluster_idx == 0
            # Create a new cluster
            push!(clusters, [token])
        else
            # Add to existing cluster
            push!(clusters[cluster_idx], token)
        end
    end
    
    # Build micro-models from clusters
    for cluster in clusters
        # Filter out any tokens that might have been removed
        valid_tokens = filter(token -> haskey(module_obj.tokens, token), cluster)
        
        # Skip if no valid tokens remain
        if isempty(valid_tokens)
            continue
        end
        
        # Sort tokens by importance
        sorted_cluster = sort(valid_tokens, by=x->module_obj.tokens[x].importance, rev=true)
        
        # Add as a micro-model
        importance = sum(module_obj.tokens[token].importance for token in sorted_cluster) / length(sorted_cluster)
        push!(module_obj.micro_models, MicroModel(sorted_cluster, importance))
    end
end

"""
    merge_tokens(token1::String, token2::String, module_name::String)

Merge two tokens in a module, keeping the more important one.
"""
function merge_tokens(token1::String, token2::String, module_name::String)
    # Get module
    if !haskey(modules, module_name)
        error("Module not found: $module_name")
    end
    
    module_obj = modules[module_name]
    
    # Skip if either token doesn't exist
    if !haskey(module_obj.tokens, token1) || !haskey(module_obj.tokens, token2)
        return
    end
    
    # Keep the more important token
    if module_obj.tokens[token1].importance >= module_obj.tokens[token2].importance
        # Merge token2 into token1
        module_obj.tokens[token1].frequency += module_obj.tokens[token2].frequency
        module_obj.tokens[token1].importance = (module_obj.tokens[token1].importance + module_obj.tokens[token2].importance) / 2
        
        # Merge connections
        for (conn, strength) in module_obj.tokens[token2].connections
            if haskey(module_obj.tokens[token1].connections, conn)
                module_obj.tokens[token1].connections[conn] += strength
            else
                module_obj.tokens[token1].connections[conn] = strength
            end
        end
        
        # Remove token2
        delete!(module_obj.tokens, token2)
    else
        # Merge token1 into token2
        module_obj.tokens[token2].frequency += module_obj.tokens[token1].frequency
        module_obj.tokens[token2].importance = (module_obj.tokens[token1].importance + module_obj.tokens[token2].importance) / 2
        
        # Merge connections
        for (conn, strength) in module_obj.tokens[token1].connections
            if haskey(module_obj.tokens[token2].connections, conn)
                module_obj.tokens[token2].connections[conn] += strength
            else
                module_obj.tokens[token2].connections[conn] = strength
            end
        end
        
        # Remove token1
        delete!(module_obj.tokens, token1)
    end
end

end # module TokenSystem
