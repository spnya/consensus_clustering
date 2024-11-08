% matlab_clustering.m
function consensus_matrix = matlab_clustering(partitions)
    % Number of partitions
    M = length(partitions);
    % Number of objects
    N = length(partitions{1});
    
    % Initialize consensus matrix
    consensus_matrix = zeros(N, N);
    
    % Compute consensus matrix
    for m = 1:M
        for i = 1:N
            for j = i:N
                if partitions{m}(i) == partitions{m}(j)
                    consensus_matrix(i, j) = consensus_matrix(i, j) + 1;
                    consensus_matrix(j, i) = consensus_matrix(i, j);
                end
            end
        end
    end
    consensus_matrix = consensus_matrix / M;
end

