


 
a = [1,2,3,5,7,9,0]

b = [1,6,3,2,2,2,0]

h = [if a[i]==b[i] 1 else 0 end for i in 1:length(a)]
k = count(x->x==1, h)
println(h)
print(k)

