function [X_rec,A] = STTC_CF(M_Omega, Omega, adj)

r=[15,15,15];
Data_Size = size(M_Omega);
I=Data_Size;
max_iter=500;
inner_iter=20;
tol=1e-6;
inner_tol=1e-3;
lambda=[0.1,0.1,0.1];
alpha=[1e-8,1e-8,1e-8];
rho=1.5;
dim = length(Data_Size);
scale = 10000;
epsilon=1e-50;
Omega_c=1-Omega;

% Initialization
Gtr=cell(size(Data_Size,2),1);
G_m=cell(size(Data_Size,2),1);%matrix: Gn(2)
Q_m=cell(size(Data_Size,2),1);
U_m=cell(size(Data_Size,2),1);
Z_m=cell(size(Data_Size,2),1);
V_m=cell(size(Data_Size,2),1);

for i=1:dim-1
    Gtr{i}=randn(r(i),Data_Size(i),r(i+1));
    G_m{i}=double(tenmat(Gtr{i},2));

end
Gtr{dim}=randn(r(dim),Data_Size(dim),r(1));
G_m{dim}=double(tenmat(Gtr{dim},2));

% feature matrix
L=cell(size(Data_Size,2),1);
% space
L{1}=adj;
for i=2:dim
    ctoe=[-1,1,zeros(1,Data_Size(i)-2)];
    rtoe=[-1,zeros(1,Data_Size(i)-2)];
    Toep=toeplitz(ctoe,rtoe);%Toeplitz(0, 1, -1)
    L{i}=Toep*Toep';
end

%precompute
A=zeros(Data_Size);
for j = 1 : 10
    for i=1:dim
        Gtr{i} = precompute(TensPermute(Omega, i), Gtr([i:dim, 1:i-1]),TensPermute(M_Omega, i));
    end
end

iter =1;
G_cat=reshape(Ui2U(Gtr), Data_Size);
X_rec=G_cat.*Omega_c+(M_Omega).*Omega;

while iter < max_iter

    G_pre=G_m;X_pre=X_rec;

    N=X_rec-A;

    for i=1:dim
        GtrT=Gtr([i:dim, 1:i-1]);
        [r_n, In, r_n1] = size(GtrT{1});
        C = tens2mat(TensPermute(TCP(GtrT(2:end)),2),1,2:dim); 
        N_m=tens2mat(TensPermute(N,i),1,2:dim);
        Z_m{i}=G_m{i};
        V_m{i}=Z_m{i}*0;
        Q_m{i}=G_m{i};
        U_m{i}=Q_m{i}*0;
        ni=[1e-3,1e-3,1e-3];% tuned
        mu=[1e-1,1e-1,1e-1];% tuned

        for j=1:inner_iter

            Gmi_pre=G_m{i};

            G_m{i}=(N_m*C+0.5*mu(i)*Z_m{i}-0.5*V_m{i}+0.5*ni(i)*Q_m{i}-0.5*U_m{i}+alpha(i)*G_pre{i})/(C'*C+(alpha(i)+0.5*mu(i)+0.5*ni(i))*eye(r_n*r_n1)); 
            
            Q_m{i}=(2*lambda(i)*L{i}+ni(i)*eye(I(i)))\(ni(i)*G_m{i}+U_m{i});
    
            Z=G_m{i}+V_m{i}/mu(i);
            Z(Z<0)=0;
            Z_m{i}=Z;

            U_m{i}=U_m{i}+ni(i)*(G_m{i}-Q_m{i});

            V_m{i}=V_m{i}+mu(i)*(G_m{i}-Z_m{i});

            ni(i)=min([1.5*ni(i),1e+8]);

            mu(i)=min([1.5*mu(i),1e+8]);

            d_GG=sum((G_m{i}-Gmi_pre).^2,"all")/sum((Gmi_pre).^2,"all");
            d_QG=sum((G_m{i}-Q_m{i}).^2,"all")/sum((G_m{i}).^2,"all");
            d_ZG=sqrt(sum((G_m{i}-Z_m{i}).^2,"all"))/sum((G_m{i}).^2,"all");
            s_gg=[d_GG,d_QG,d_ZG];

            if max(s_gg)<inner_tol
                break;
            end

        end

    Gtr{i}=permute(mat2tens(G_m{i},[In, r_n,r_n1],1,[2,3]),[2,1,3]);%(I_i) x (r_i r_i+1)  ---> (I_i) x (r_i) x (r_i+1) 

    end

    R = X_rec - G_cat ;
    [A,scale] = solve_CF(R,scale,epsilon);

    G_cat=reshape(Ui2U(Gtr), Data_Size);
    X_rec=G_cat.*Omega_c+M_Omega.*Omega;    

       
    d_X=sum((X_rec-X_pre).^2,"all")/sum((X_pre).^2,'all');

    if d_X > tol
        break
    end

    iter =iter+1;
    
end

end

function [A,scale] = solve_CF(R_Omega,scale,epsilon)
    n_abs=abs(R_Omega(R_Omega~=0));
    n_std=std(n_abs);
    n_sort=sort(n_abs);
    IQR = prctile(n_sort,95);
    deta2 = 1.06*min(n_std,IQR/1.34)*(length(n_abs)^-0.2);
    w = exp(-(n_abs./(deta2)));
    anomalies_idx = (w<epsilon); 
    anomalies=n_abs(anomalies_idx);
    w_isempty=isempty(anomalies);
    if ~w_isempty
        scale=min(min(anomalies),scale);
    end
    ONE_1=ones(size(R_Omega));
    ONE_1(abs(R_Omega)-scale<0)=0;
    A = R_Omega.*ONE_1;

end


