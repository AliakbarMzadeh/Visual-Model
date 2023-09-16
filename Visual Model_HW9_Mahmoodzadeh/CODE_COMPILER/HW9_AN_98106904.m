

%%  natural image
clc; clear all; close all;
rng shuffle
addpath('/Users/aliakbarmahmoodzadeh/Desktop/HW_9_ADN/sparsenet')
load IMAGES.mat

montage(cell2mat(x),'size',[1,1])


%%
% Load the images
load('IMAGES.mat');

% Get the number of images
num_images = size(IMAGES, 3);

% Initialize an empty array to store images
x = zeros(size(IMAGES));

% Populate the array with images
for i = 1:num_images
    x(:,:,i) = IMAGES(:,:,i);
end

% Display the images
montage(x, 'Size', [1, 1]);
%  sparsenet
%sparsenet


%%
% Load the MNIST data
data = load('mnist.mat');

% Extract images (assuming 'trainX' is the field containing images)
IMAGES = data.trainX;

% Get the number of images
num_images = size(IMAGES, 1);

% Get the size of one image (assuming images are square)
image_size = sqrt(size(IMAGES, 2));

% Initialize an empty array to store images
x = zeros(image_size, image_size, num_images);

% Populate the array with images
for i = 1:num_images
    % Reshape each row into an image and store it in the array
    x(:,:,i) = reshape(IMAGES(i,:), [image_size, image_size]);
end

% Display the images
montage(x, 'Size', [5, 10]);  % Adjust the 'Size' parameter to change the montage layout










%%

%%
% sparsenet.m - simulates the sparse coding algorithm
% 
% Before running you must first define A and load IMAGES.
% See the README file for further instructions.
addpath('/Users/aliakbarmahmoodzadeh/Desktop/HW_9_ADN/sparsenet')
load IMAGES.mat
batch_size= 100;

num_images=size(IMAGES,3);
image_size=size(IMAGES,1);
BUFF=4;

[L M]=size(A);
sz=sqrt(L);

eta = 1.0;
noise_var= 0.01;
beta= 2.2;
sigma=0.316;
tol=.01;

VAR_GOAL=0.1;
S_var=VAR_GOAL*ones(M,1);
var_eta=.001;
alpha=.02;
gain=sqrt(sum(A.*A))';

X=zeros(L,batch_size);

display_every=10;

h=display_network(A,S_var);

% main loop

for image_no=1:size(IMAGES,3)
    t = IMAGES(:,:,image_no);
    IMAGES(:,:,image_no) = IMAGES(:,:,image_no)/sqrt(var(t(:))/VAR_GOAL);
end

for t=1:num_trials

    % choose an image for this batch
    
    i=ceil(num_images*rand);
    this_image=IMAGES(:,:,i);
    this_slaiencyMap = saliencyMaps(:,:,i);
    this_slaiencyMap = this_slaiencyMap > 3*mean(this_slaiencyMap,'all');
    [row_no0, col_no0] = find(this_slaiencyMap>0);
    % extract subimages at random from this image to make data vector X
    
    for i=1:batch_size
        if rand < 0.5
            r=BUFF+ceil((image_size-sz-2*BUFF)*rand);
            c=BUFF+ceil((image_size-sz-2*BUFF)*rand);
        else
            rand_index = randi(length(row_no0), 1, 1);
            r = row_no0(rand_index);
            c = col_no0(rand_index);
        end
        X(:,i)=reshape(this_image(r:r+sz-1,c:c+sz-1),L,1);
    end
    
    % calculate coefficients for these data via conjugate gradient routine
    
    S=cgf_fitS(A,X,noise_var,beta,sigma,tol);
    
    % calculate residual error
    
    E=X-A*S;
    
    % update bases
    
    dA=zeros(L,M);
    for i=1:batch_size
        dA = dA + E(:,i)*S(:,i)';
    end
    dA = dA/batch_size;
    
    A = A + eta*dA;
    
    % normalize bases to match desired output variance
    
    for i=1:batch_size
        S_var = (1-var_eta)*S_var + var_eta*S(:,i).*S(:,i);
    end
    gain = gain .* ((S_var/VAR_GOAL).^alpha);
    normA=sqrt(sum(A.*A));
    for i=1:M
        A(:,i)=gain(i)*A(:,i)/normA(i);
    end
    
    % display
    
    if (mod(t,display_every)==0)
        display_network(A,S_var,h);
    end
    
end

%%
% sparsenet.m - simulates the sparse coding algorithm
% 
% Before running you must first define A and load IMAGES.
% See the README file for further instructions.
addpath('/Users/aliakbarmahmoodzadeh/Desktop/HW_9_ADN/sparsenet')
load IMAGES.mat

batch_size= 100;

num_images=size(IMAGES,3);
image_size=size(IMAGES,1);
BUFF=4;

[L M]=size(A);
sz=sqrt(L);

eta = 1.0;
noise_var= 0.01;
beta= 2.2;
sigma=0.316;
tol=.01;

VAR_GOAL=0.1;
S_var=VAR_GOAL*ones(M,1);
var_eta=.001;
alpha=.02;
gain=sqrt(sum(A.*A))';

X=zeros(L,batch_size);

display_every=10;

h=display_network(A,S_var);

% main loop

for image_no=1:size(IMAGES,3)
    t = IMAGES(:,:,image_no);
    IMAGES(:,:,image_no) = IMAGES(:,:,image_no)/sqrt(var(t(:))/VAR_GOAL);
end

num_trials = 100
for t=1:num_trials

    % choose an image for this batch
    
    i=ceil(num_images*rand);
    this_image=IMAGES(:,:,i);
    
    % extract subimages at random from this image to make data vector X
    
    for i=1:batch_size
        r=BUFF+ceil((image_size-sz-2*BUFF)*rand);
        c=BUFF+ceil((image_size-sz-2*BUFF)*rand);
        X(:,i)=reshape(this_image(r:r+sz-1,c:c+sz-1),L,1);
    end
    
    % calculate coefficients for these data via conjugate gradient routine
    
    S=cgf_fitS(A,X,noise_var,beta,sigma,tol);
    
    % calculate residual error
    
    E=X-A*S;
    
    % update bases
    
    dA=zeros(L,M);
    for i=1:batch_size
        dA = dA + E(:,i)*S(:,i)';
    end
    dA = dA/batch_size;
    
    A = A + eta*dA;
    
    % normalize bases to match desired output variance
    
    for i=1:batch_size
        S_var = (1-var_eta)*S_var + var_eta*S(:,i).*S(:,i);
    end
    gain = gain .* ((S_var/VAR_GOAL).^alpha);
    normA=sqrt(sum(A.*A));
    for i=1:M
        A(:,i)=gain(i)*A(:,i)/normA(i);
    end
    
    % display
    
    if (mod(t,display_every)==0)
        display_network(A,S_var,h);
    end
    
end


%%
% sparsenet.m - simulates the sparse coding algorithm
% 
% Before running you must first define A and load IMAGES.
% See the README file for further instructions.

clc; clear; close all;

A = rand(576,64)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));

%%%  natural images
% IMAGES = load('IMAGES.mat');
% IMAGES = IMAGES.IMAGES;
% IMAGES = IMAGES(:,:,randperm(size(IMAGES,3),10)); % randomly select 10 images

%%% yale images
% IMAGES = load('All_YALE_dewhited');
% IMAGES = IMAGES.IMAGES;
% IMAGES = IMAGES(:,:,randperm(size(IMAGES,3),10)); % randomly select 10 images

%%% MNIST images
% IMAGES = load('All_MNIST_dewhited');
% IMAGES = IMAGES.IMAGES;
% IMAGES = IMAGES(:,:,randperm(size(IMAGES,3),10)); % randomly select 10 images

%%% Caltech101 images
% IMAGES = load('All_Caltech_dewhited');
% IMAGES = IMAGES.IMAGES;
% IMAGES = IMAGES(:,:,randperm(size(IMAGES,3),10)); % randomly select 10 images

%%% BIRD Video Frames
[rgb_vid, video_patches] = import_process_video();

% cal sparse functions for the first patch
first_video_patch = video_patches(:,:,1:10);
IMAGES = first_video_patch;

% imshow selected images
figure;
for i = 1:size(IMAGES,3)
    subplot(2,5,i);
    imshow(IMAGES(:,:,i));
    title("IMAGE " + i,'interpreter','latex');
end

num_trials = 10000;
batch_size = 100;

num_images=size(IMAGES,3);
image_size=size(IMAGES,1);
BUFF = 4;

[L M] = size(A);
sz = sqrt(L);

eta = 1.0;
noise_var = 0.01;
beta = 2.2;
sigma =0.316;
tol =.01;

VAR_GOAL = 0.1;
S_var = VAR_GOAL*ones(M,1);
var_eta = .001;
alpha = .02;
gain = sqrt(sum(A.*A));

X = zeros(L,batch_size);

display_every = 10;

h = display_network(A,S_var);

% main loop
for t = 1:num_trials

    % choose an image for this batch
    i = ceil(num_images*rand);
    this_image = IMAGES(:,:,i);
    if(length(find(isnan(this_image) == 1)) == 0)
        % extract subimages at random from this image to make data vector X
        for i = 1:batch_size
            r = BUFF+ceil((image_size-sz-2*BUFF)*rand);
            c = BUFF+ceil((image_size-sz-2*BUFF)*rand);
            X(:,i) = reshape(this_image(r:r+sz-1,c:c+sz-1),L,1);
        end

        % calculate coefficients for these data via conjugate gradient routine
        S = cgf_fitS(A,X,noise_var,beta,sigma,tol);

        % calculate residual error
        E = X-A*S;

        % update bases
        dA = zeros(L,M);

        for i = 1:batch_size
            dA = dA + E(:,i)*S(:,i)';
        end

        dA = dA/batch_size;
        A = A + eta*dA;

        % normalize bases to match desired output variance
        for i=1:batch_size
            S_var = (1-var_eta)*S_var + var_eta*S(:,i).*S(:,i);
        end
        gain = gain .* ((S_var/VAR_GOAL).^alpha);
        normA = sqrt(sum(A.*A));
        for i = 1:M
            A(:,i) = gain(i)*A(:,i)/normA(i);
        end

        % display
        if (mod(t,display_every) == 0)
            display_network(A,S_var,h);
        end
    end
end

%%
clc; clear;

% Load the video file
v = VideoReader('BIRD.avi');

% Parameters
patchSize = [8, 8];  % Size of the patches to extract
dictionarySize = 100; % Number of atoms in the dictionary
windowToInspect = 60; % Window number to be inspected

% Read frames and convert them to grayscale
frames = [];
while hasFrame(v)
    frame = readFrame(v);
    frame = rgb2gray(frame);
    frames = cat(3, frames, frame);
end

% Collect patches from the video frames
patches = [];
for i = 1:size(frames, 3)
    patches = [patches, im2col(frames(:, :, i), patchSize, 'distinct')];
end

% Use K-SVD to learn a dictionary and sparse code the patches
params.data = patches;
params.Tdata = 10; % Sparsity level
params.dictsize = dictionarySize;
params.iternum = 20;
params.memusage = 'high';
[D, X] = KSVD(params,'');

% Now, let's create a plot of coefficient values vs frame number
sparseCoefficients = zeros(size(X, 2), size(frames, 3));
for i = 1:size(frames, 3)
    framePatches = im2col(frames(:, :, i), patchSize, 'distinct');
    sparseCoefficients(:, i) = full(sum(X(:, patches == framePatches), 2));
end
figure; plot(sparseCoefficients');
xlabel('Frame number');
ylabel('Sparse coefficient value');
title('Sparse coefficient values vs frame number');

% Now, let's create a plot of coefficient values through time for a specific window
figure; plot(sparseCoefficients(windowToInspect, :)');
xlabel('Frame number');
ylabel('Sparse coefficient value');
title(['Sparse coefficient values through time for window ', num2str(windowToInspect)]);

%%

% Load the video file
v = VideoReader('BIRD.avi');

% Parameters
patchSize = [8, 8];  % Size of the patches to extract
dictSize = 100; % Number of atoms in the dictionary
sparsity = 10;  % Desired sparsity level

% Read frames and convert them to grayscale
frames = [];
while hasFrame(v)
    frame = readFrame(v);
    frame = rgb2gray(frame);
    frame = double(frame); % convert the frame to double
    frames = cat(3, frames, frame);
end

% Collect patches from the video frames
patches = [];
for i = 1:size(frames, 3)
    patches = [patches, im2col(frames(:, :, i), patchSize, 'distinct')];
end

nDim = prod(patchSize); % Number of dimensions for each atom

% Create a random initial dictionary
D = randn(nDim, dictSize);
D = bsxfun(@rdivide, D, sqrt(sum(D.^2, 1)));

% Learn the dictionary and compute sparse coefficients using MOD-OMP
maxIter = 20;
for iter = 1:maxIter
    % Sparse coding step: compute sparse coefficients using OMP
    X = zeros(dictSize, size(patches, 2));
    for i = 1:size(patches, 2)
        X(:, i) = omp(D, patches(:, i), sparsity);
    end

    % Dictionary update step: use Method of Optimal Directions
    D = patches * X' / (X * X');

    % Normalize the dictionary
    D = bsxfun(@rdivide, D, sqrt(sum(D.^2, 1)));
end

% Sparse coding step: compute sparse coefficients for all patches
X = zeros(dictSize, size(patches, 2));
for i = 1:size(patches, 2)
    X(:, i) = omp(D, patches(:, i), sparsity);
end















%% Functions

function sparse_coding_on_video
    % Load the video file
    v = VideoReader('BIRD.avi');

    % Parameters
    patchSize = [8, 8];  % Size of the patches to extract
    dictSize = 100; % Number of atoms in the dictionary
    sparsity = 10;  % Desired sparsity level

    % Read frames and convert them to grayscale
    frames = [];
    while hasFrame(v)
        frame = readFrame(v);
        frame = rgb2gray(frame);
        frame = double(frame); % convert the frame to double
        frames = cat(3, frames, frame);
    end

    % Collect patches from the video frames
    patches = [];
    for i = 1:size(frames, 3)
        patches = [patches, im2col(frames(:, :, i), patchSize, 'distinct')];
    end

    nDim = prod(patchSize); % Number of dimensions for each atom

    % Create a random initial dictionary
    D = randn(nDim, dictSize);
    D = bsxfun(@rdivide, D, sqrt(sum(D.^2, 1)));

    % Learn the dictionary and compute sparse coefficients using MOD-OMP
    maxIter = 20;
    for iter = 1:maxIter
        % Sparse coding step: compute sparse coefficients using OMP
        X = zeros(dictSize, size(patches, 2));
        for i = 1:size(patches, 2)
            X(:, i) = omp(D, patches(:, i), sparsity);
        end

        % Dictionary update step: use Method of Optimal Directions
        D = patches * X' / (X * X');

        % Normalize the dictionary
        D = bsxfun(@rdivide, D, sqrt(sum(D.^2, 1)));
    end

    % Sparse coding step: compute sparse coefficients for all patches
    X = zeros(dictSize, size(patches, 2));
    for i = 1:size(patches, 2)
        X(:, i) = omp(D, patches(:, i), sparsity);
    end
end

% Define the omp function
function x = omp(D, y, sparsity)
    residual = y;
    index = [];
    for i = 1:sparsity
        dot_products = abs(D' * residual);
        [~, idx] = max(dot_products);
        index = [index, idx];
        x_temp = D(:, index) \ y;
        residual = y - D(:, index) * x_temp;
    end
    x = zeros(size(D, 2), 1);
    x(index) = x_temp;
end




function IMAGES = whitening(images)
    N = size(images,1);
    imgNum = size(images,3);
    M = imgNum;
    [fx, fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
    rho=sqrt(fx.*fx+fy.*fy);
    f_0=0.4*N;
    filt=rho.*exp(-(rho/f_0).^4);
    for i=1:M
        image=images(:,:,i);
        If=fft2(image);
        imagew=real(ifft2(If.*fftshift(filt)));
        IMAGES(:,:,i) = imagew;
    end
    IMAGES=sqrt(0.1)*IMAGES./sqrt(mean(var(IMAGES)));
end


% If you want to use some other images, there are a number of
% preprocessing steps you need to consider beforehand.  First, you
% should make sure all images have approximately the same overall
% contrast.  One way of doing this is to normalize each image so that
% the variance of the pixels is the same (i.e., 1).  Then you will need
% to prewhiten the images.  For a full explanation of whitening see
% 
%   Olshausen BA, Field DJ (1997)  Sparse Coding with an Overcomplete
%   Basis Set: A Strategy Employed by V1?  Vision Research, 37: 3311-3325. 
% 
% Basically, to whiten an image of size NxN, you multiply by the filter
% f*exp(-(f/f_0)^4) in the frequency domain, where f_0=0.4*N (f_0 is the
% cutoff frequency of a lowpass filter that is combined with the
% whitening filter).  Once you have preprocessed a number of images this
% way, all the same size, then you should combine them into one big N^2
% x M array, where M is the number of images.  Then rescale this array
% so that the average image variance is 0.1 (the parameters in sparsenet
% are set assuming that the data variance is 0.1).  Name this array
% IMAGES, save it to a file for future use, and you should be off and
% running.  The following Matlab commands should do this:


function make_your_own_images(input_images,name)
    image_size = size(input_images,1);
    num_images = size(input_images,3);
    N = image_size;
    M = num_images;

    [fx fy] = meshgrid(-N/2:N/2-1,-N/2:N/2-1);
    rho = sqrt(fx.*fx+fy.*fy);
    f_0 = 0.4*N;
    filt = rho.*exp(-(rho/f_0).^4);

    for i = 1:M
        image = input_images(:,:,i);  % you will need to provide get_image
        If = fft2(image);
        imagew = real(ifft2(If.*fftshift(filt)));
        IMAGES(:,i) = reshape(imagew,N^2,1);
    end
    
    IMAGES = sqrt(0.1)*IMAGES/sqrt(mean(var(IMAGES)));
    IMAGES = reshape(IMAGES,[image_size image_size num_images]);
    IMAGES = 0.1./var(IMAGES,0,[1 2]).*IMAGES; % set var to var goal = 0.1
    save(name,'IMAGES')
end

function [rgb_vid, patches] = import_process_video()
    % load the video
    v = VideoReader('BIRD.avi');
    v_frames_num = v.Duration*v.FrameRate;
    
    % export the video frames
    channels_num = 1; % will be converted to grayscale frames
    frame = zeros(v.Height,v.Width,channels_num,v_frames_num);
    frame_rgb = zeros(v.Height,v.Width,3,v_frames_num);
    
    im_size = size(frame,1);
    
    k = 1;
    while hasFrame(v)
        frame_rgb(:,:,:,k) = readFrame(v);
        frame(:,:,:,k) = im2gray(uint8(frame_rgb(:,:,:,k))); 
        k = k+1;
    end
    
    frame = squeeze(frame);
    
    % resize images
    frame = frame(1:im_size,1:im_size,:);
    rgb_vid = frame_rgb(1:im_size,1:im_size,:,:);
    rgb_vid = uint8(rgb_vid);
    
    % remove white noise from frames
    make_your_own_images(frame,'Bird_Frames_dewhited');
    frame = load('Bird_Frames_dewhited.mat').IMAGES;
    

    v_frames_num = size(frame,3);
    patch_size = 10;
    patches = frame;
    
end

