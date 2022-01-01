using Random

function train_test_split(data, y_column, train_size)
    #set seed = 30 -> fix random state
    Random.seed!(30);
    n = size(data,1)
    #random
    idx = shuffle(Vector(1:n))
    #split train and test dataset
    train_idx = view(idx, 1:floor(Int, train_size*n))
    test_idx = view(idx, (floor(Int, train_size*n)+1):n)
    data[train_idx, :], y_column[train_idx, :],  data[test_idx, :], y_column[test_idx, :]
end


function read_file(filename)
    X=[]
    y = []
    fp = open(filename,"r")
    first_line = readline(fp)
    attr = filter!(e->e != "Id",split(first_line,","))
    while (!eof(fp))
        line = readline(fp)
        x_line = split(line,",")
        popfirst!(x_line)
        y_line = pop!(x_line)
        x_line = [parse(Float64,ss) for ss in x_line]

        #create x
        append!(X, [x_line])
        
        #create y
        push!(y, y_line)
    end
    X = hcat(X...)'
    close(fp)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42);
    X_train, y_train, X_test, y_test = train_test_split(X, y,2/3)
    return X_train, y_train, X_test, y_test, attr
end

mutable struct Node
    name::String
    child::Array{Node,1}
    entropy::Float64
    pos::Array{Int,1}
    depth::Int
    label::String
    value_split::Float64

    function Node(name, entropy, pos, depth)
        child = Array{Node,1}(undef, 0)
        nd = new(name,child,entropy,pos,depth)
        nd
    end

    function Node(name,child,entropy,pos,depth)
        nd = new(name,child,entropy,pos,depth)
        return nd
    end
end


mutable struct DecisionTree
    min_value_gain::Int
    depth::Int
    attr
    root::Node

    function DecisionTree(min_value_gain, depth, attr)
        tree = new(min_value_gain,depth,attr)
        return tree
    end
end

function fit(X_train, y_train, tree, attr)
    label = Set(y_train)

    #calc entrophy
    n = [length(y_train)]
    y_frequent = [count(==(element),y_train) for element in label]
    prob = broadcast(/,y_frequent,n)
    entropy = sum(-broadcast(*,prob,log2.(prob)))
    
    #set root
    pos = [i for i in 1:length(X_train[:,1])]
    root = Node("",entropy,pos,0)

    #init tree
    tree.root = root
    list_root_check = [root]
    #process with tree's child
    while length(list_root_check) != 0
        active_node = pop!(list_root_check)
        if active_node.entropy < tree.min_value_gain || active_node.depth < tree.depth
            active_node.child = make_split(active_node, X_train, y_train)
            if length(active_node.child) == 0
                y_dummy = [y_train[i] for i in active_node.pos]
                vals_unique = unique(y_dummy)
                count_vals_unique = [count(==(element),y_dummy) for element in vals_unique]
                frequent_value = findmax(count_vals_unique)[2]
                active_node.label = vals_unique[frequent_value]
                active_node.name = vals_unique[frequent_value]
            end
            append!(list_root_check, active_node.child)
        else
            y_dummy = [y_train[i] for i in active_node.pos]
            vals_unique = unique(y_dummy)
            count_vals_unique = [count(==(element),y_dummy) for element in vals_unique]
            frequent_value = findmax(count_vals_unique)[2]
            active_node.label = vals_unique[frequent_value]
            active_node.name = vals_unique[frequent_value]
        end
    
    end
    return tree
end

function make_split(node::Node, x_train, y_train)
    choose_gain = 0
    choose_split = []
    choose_attr = ""
    best_value = 0
    child_nodes = Array{Node,1}(undef, 0)
    entropy_best = []
    x_train_T = copy(x_train)'
    pos = node.pos
    for col in 1:length(x_train_T[:,1])
        H_min = 10
        left_set_choose = []
        right_set_choose = []
        entropy_list_choose = []
        value_choose = 0

        #get unique value to choose cutoff
        unique_value = Set(sort(x_train_T[col,:]))
        if length(unique_value) == 1
            continue
        end

        dummy = copy(x_train_T[col,:])
        dummy = [if i ∉ pos 0 else dummy[i] end for i in 1:length(dummy)]
        
        for value in unique_value
            H = 0
            entropy_list = []
            
            # left_set = findall(<(value), x_train_T[col,:])
            # right_set = findall(>(value), x_train_T[col,:])
            left_set = findall(x->(x<=value) && (x!=0), dummy)
            right_set = findall(x->(x>value) && (x!=0) , dummy)
            
            for set in [left_set,right_set] 
                value_cor_y = [y_train[i] for i in set]
                value_statisfy = [count(==(element),value_cor_y) for element in Set(y_train)]
                if count(x->x==0, value_statisfy) == length(value_statisfy)
                    entropy = 0
                else
                    value_statisfy = [if value_statisfy[i]==0 0.00001 else value_statisfy[i] end for i in 1:length(value_statisfy)]
                    prob = broadcast(/,value_statisfy,[length(set)])
                    entropy = sum(-broadcast(*,prob,log2.(prob)))
                end                       
                    append!(entropy_list, entropy)
                
                H += (length(set)/length(x_train_T[col,:]))*entropy
                if count(x->x==0, entropy_list) == 2
                    print("x")
                end
            end
            #choose cutoff for each column
            if H < H_min 
                H_min = H
                left_set_choose = left_set
                right_set_choose = right_set
                value_choose = value
                entropy_list_choose = entropy_list
            end
        end
        if minimum([length(left_set_choose), length(right_set_choose)]) < 2
            continue
        end
        
        #calc information gain
        gain_information = node.entropy - H_min
        #choose column which have a higher information gain
        if gain_information > choose_gain
            choose_gain = gain_information
            choose_attr = col
            choose_split = [left_set_choose,right_set_choose]
            best_value = value_choose
            entropy_best = entropy_list_choose
        end
    end
    node.name = string(choose_attr)
    node.value_split = best_value
    k = 1
    #create new child
    for split in choose_split
        if k == 1
            new_node = Node("< "*string(best_value),entropy_best[k],split,node.depth+1)
            push!(child_nodes, new_node)
        else
            new_node = Node(">"*string(best_value),entropy_best[k],split,node.depth+1)
            push!(child_nodes, new_node)
        end
        k += 1
    end
    return child_nodes
end


function predict(tree::DecisionTree, x_test)
    number_data_point =  length(x_test[:,1])
    labels_arr = Vector{Union{String, Nothing}}(undef, number_data_point)
    fill!(labels_arr, nothing)

    for n in 1:number_data_point
        x = x_test[n,:]
        node = deepcopy(tree.root)
        while length(node.child) != 0
            value = x[parse(Int64,node.name)]
            if value <= node.value_split
                node = node.child[1]
            else
                node = node.child[2]
            end
        end
        labels_arr[n] = node.label
    end
    return labels_arr
end

function accuracy(y_predict, y_true)
    correct_arr = [if y_predict[i]==y_true[i] 1 else 0 end for i in 1:length(y_predict)]
    return count(x->x==1, correct_arr) / length(y_true)
end

function main()
    X_train, y_train, X_test, y_test, attr = read_file(joinpath(@__DIR__,"Iris.csv"))
    tree = DecisionTree(0, 10, attr)
    fit(X_train, y_train, tree, attr)
    y_hat_train = predict(tree, X_train)
    acc_train = accuracy(y_hat_train, y_train)

    y_hat_test = predict(tree, X_test)
    acc_test = accuracy(y_hat_test, y_test)
    
    println("Accuracy of train dataset: ",acc_train,"\n")
    println("Accuracy of test dataset: ",acc_test,"\n")
end

main()


