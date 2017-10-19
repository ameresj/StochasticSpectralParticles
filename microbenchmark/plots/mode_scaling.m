
%% Results of the benchmark


f90_Nf=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096];
f90_time=[ 2.90514612,3.01999521,2.86137962,3.15903354,4.18794298 , ....
            5.52950048 , 8.91208267 , 18.2969398 , 32.0790787, ....
                          56.6839371,113.527840,227.305740,453.329163 ];
                  
                      
julia_Nf=[2, 4,8,16, 32,64,128,256,512,1024,2048,4096];
julia_time=[3.855679911, 4.042620154, 4.363989855, 5.159452995,...
                 7.001579742 , 10.58972716, 17.570501146,30.31221257,...
                 55.095773526,107.5852229,210.180210535,417.013088662];
             
             
%Opencl  CPU
opencl_cpu_Nf=[   1    2    4    8   16   32   64  128  256  512 1024 2048 4096 8192];
opencl_cpu_time=[ 6.80458999    6.32910395    6.91702509    6.63927412    6.78454494...
     7.17289901   10.15140486   14.62904382   20.80181789   36.36334395...
    70.78291297  134.45345283  266.84051013  518.608706  ];
opencl_cpu_time_accum=[   3.47659373    3.30743575    3.71662927    3.66902876    3.77672338...
     4.01219702    6.42026782   10.38936615   15.89970684   29.55536866...
    59.55449438  115.96323323  232.97765827  455.74415135];
opencl_cpu_time_proj=[  2.29527521   2.17301726   2.27766228   2.15295815   2.19670892...
    2.32477474   2.79886985   3.27097845   3.99533677   5.86442137...
   10.19845057  17.47412801  32.72724867  61.74806833];
%opencl GPU
opencl_gpu_time=[   3.09054279    3.27073812    3.42586017    3.92009282    6.51692486...
     6.80322003   11.29690385   18.88198185   41.58200192   79.98651695...
   158.05113482];
opencl_gpu_Nf=[   1    2    4    8   16   32   64  128  256  512 1024];
opencl_gpu_time_accum=[   0.17679691    0.31838179    0.64977145    1.07466173    1.91727614...
     3.82188988    7.98902607   15.06871796   35.94886613   71.54035807...
   143.91334057];
opencl_gpu_time_proj=[  2.11471915   2.19467878   2.17454362   2.26884174   4.01441169...
    2.37615061   2.65939379   3.18228674   4.85318232   7.66778708...
   13.35461736];

set(0,'DefaultLegendFontSize',16,...
    'DefaultLegendFontSizeMode','manual',...
    'DefaultAxesFontSize', 14)
set(0,'DefaultLineMarkerSize', 6);
set(0,'defaultLineLineWidth', 2);

figure;
loglog(f90_Nf,f90_time,'k-o'); hold on;
loglog(julia_Nf,julia_time,'-x');
loglog(opencl_cpu_Nf,opencl_cpu_time,'-o');
loglog(opencl_gpu_Nf,opencl_gpu_time,'-.')

xlabel('number of Fourier modes')
ylabel('wall time [s]')
grid on;
legend('gfortran(OMP)','julia(MPI)','pyOpenCL (CPU)','pyOpenCL (GPU)','Location','NorthWest')
axis tight;

print('-dpng','optimized_walltime.png');


figure;
loglog(opencl_cpu_Nf,opencl_cpu_time_accum,'k-x'); hold on;
loglog(opencl_cpu_Nf,opencl_cpu_time_proj,'k-o')
 loglog(opencl_gpu_Nf,opencl_gpu_time_accum,'b-x'); hold on;
 loglog(opencl_gpu_Nf,opencl_gpu_time_proj,'b-o')
ylabel('wall time [s]')
xlabel('number of Fourier modes')
grid on; 
legend('accumulation (CPU)','projection(CPU)',...
    'accumulation (GPU)','projection(GPU)', 'Location','NorthWest')
print('-dpng','kernels_opencl.png');
% axis tight;



% Quad core but without optimization of exp



%without product
opencl0_cpu_Nf=[  1   2   4   8  16  32  64 128 256 512];
opencl0_cpu_time=[  6.28354502   6.14189911   6.64169002   7.06727195   7.77858305...
    8.81784511  11.96826911  18.97678781  32.73949695  62.36309099];

opencl0_gpu_Nf=[  1   2   4   8  16  32  64 128 256 512];
opencl0_gpu_time=[   3.30970597    3.30631399    3.82307601    5.31349111    6.65852118...
     9.05740905   14.17948985   26.45427608   54.40208793  105.93067384];

f90_std_Nf=[1,2,4,8,16,32,64,128,256,512];
f90_std_time=[ 2.78780222, 5.09403610, 9.30106926 ,16.7028885 ,33.9309425,...
                65.4182358,134.019150,269.538818, 525.344543,1031.46960  ];    
figure;
loglog(f90_std_Nf,f90_std_time,'k-o');hold on;
plot(opencl0_cpu_Nf,opencl0_cpu_time,'-o')
plot(opencl0_gpu_Nf,opencl0_gpu_time,'-x'); 
xlabel('number of Fourier modes')
ylabel('wall time [s]')
grid on;
legend('gfortran(OMP)','pyOpenCL (CPU)','pyOpenCL (GPU)','Location','NorthWest')
axis tight;
print('-dpng','standard_walltime.png');






%% Programmtage
figure('Position', [100,100, 900, 350]);
loglog(f90_std_Nf,f90_std_time,'k-');hold on;
loglog(f90_Nf,f90_time,'k-o'); hold on;
loglog(julia_Nf,julia_time,'-x');
loglog(opencl_cpu_Nf,opencl_cpu_time,'*-');
loglog(opencl_gpu_Nf,opencl_gpu_time,'x-')
% plot(opencl0_cpu_Nf,opencl0_cpu_time,'*-')
% plot(opencl0_gpu_Nf,opencl0_gpu_time,'x-'); 
% pbaspect([2 1 1])
axis tight;
grid on


% figure;

xlabel('number of Fourier modes')
ylabel('wall time [s]')
legend('gfortran(OMP) a)','gfortran(OMP) b)', 'julia (MPI) b)', ...
     'pyOpenCL (CPU) a)','pyOpenCL (GPU) a)','Location','NorthWest')
print('-dpng','methods_walltime.png');




% 
% 
% % Estimate Flops
% N=1;
% figure
% plot(f90_Nf(N:end),f90_time(N:end))
% 
% P = polyfit(f90_Nf(N:end),f90_time(N:end),1);
% 
% 
% Np=1e5; Nt=150;
% flop_mode=Np*Nt/P(1)*
% 
% 
% 
% 
% flops_max=12.17e9;
% 
% 

%Number of computers	Avg. cores/computer	GFLOPS/core	GFLOPs/computer
% cpu family      : 6
% model           : 78
% model name      : Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz
% stepping        : 3
% ntel(R) Core(TM) i5-6300U CPU @ 2.40GHz [Family 6 Model 78 Stepping 3]	71	3.87	3.14	12.17

            
            