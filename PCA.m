
covariance_matrix = corr(train_data);
[eigen_vectors,eigen_values_diagonal] = eig(covariance_matrix);
eigen_values_row = eig(covariance_matrix);

[sorted_eigen_values,sorted_eigen_values_index] = sort(eigen_values_row);
eigen_vectors = eigen_vectors(:,sorted_eigen_values_index);
vectors_for_pca = eigen_vectors(:,1:50);

train_data = train_data * vectors_for_pca;
test_data = test_data * vectors_for_pca;

training_labels = train_label;
y = train_data;
k = 5;
Labels = [];
[m,n] = size(test_data);


[m_training_set,n_training_set] = size(train_data);

totalGroups=5;
groupSize=size(train_data,1)/totalGroups;
randOrder=randperm(totalGroups);
g = randsample(1:5000,5000,false);

index_matrix=reshape(g,groupSize,totalGroups)';


k_set = [ 1,3,5,7,9,11,13,15,17];
final_acc = [];
for j=1:size(k_set,2)

    acc = 0;

    for i=1:5

        test_x = train_data(index_matrix(i,:), :);
        test_y = train_label(index_matrix(i,:));
        train_x = train_data(:,:);
        train_y = train_label(:,:);
        train_x(index_matrix(i,:),:) = [];
        train_y(index_matrix(i,:)) = [];

        acc = acc + knn(train_x,train_y,test_x,test_y,k_set(j));

    end

    acc = acc / 5;
    final_acc(j) = acc;
end

final_acc
figure()
plot(k_set,final_acc)
xlabel("K")
ylabel("Accuracy")






function acc = knn(train_x,train_y,test_x,test_y,k)

m = size(test_x,1);
for i = 1:m
    Labels(i) = calculateLabel(test_x(i,:),train_x,k,train_y);
end   

Labels = Labels';
temp = test_y - Labels;
temp = nnz(temp);
acc = 1 - temp / size(Labels,1);

end
 
 


function label = calculateLabel(x,y,k, training_labels)

temp = y - x;
temp = temp.^2;
eucledian_distance = sum(temp,2);
dist_label = [eucledian_distance,training_labels];
sorted_dist_label = sortrows(dist_label,1);
first_k_labels = sorted_dist_label(1:k, 2);

x = [0 1];
[freq,indexes] = hist(first_k_labels,x);

if freq(1) > freq(2) 
    label = 0;
else 
    label = 1;
end
end









