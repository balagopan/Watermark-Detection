
clc; clear; close all;

%% === 1. Read and convert image ===
I = im2double(rgb2gray(imread('Peppers_wm.jpg')));

num_iter=2;
kappa=5;
lambda=0.05;
% I=anisoDiff(I, num_iter, kappa, lambda);

%% === 2. TV-L1 PARAMETERS (paper) ===
lambda = 0.05;
tau = 0.2;
iterations = 200;

%% === 3. TV-L1 Structure Extraction ===
S = TVL1_denoise(I, lambda, tau, iterations);

%% Step 3: Enhance structure image
S_adj = imadjust(S);         % contrast stretching
h = [0 -1 0; -1 5 -1; 0 -1 0];
S_sharp = imfilter(S_adj, h,'replicate');
% S_sharp= imsharpen(S_adj);  % sharpening

%% === 4. Canny Edge Detection (on structure image) ===
edges = edge(S_sharp, 'canny',[0.05 0.67]);

%% === 5. PARAMETERS FROM THE PAPER ===
k = 75;      % block size (odd number) - can be 33, 41 etc.
r = 102;      % tolerance range from paper

% Convert to 0–255 range (IMPORTANT for Otsu + r)
% S = S * 255;
S = S_sharp * 255;
half = floor(k/2);
[H,W] = size(S);

watermark_mask = false(H,W);

%% === 6. Block-wise Otsu + Edge Discrimination ===
for i = 1:H
    for j = 1:W

        if ~edges(i,j)
            continue;
        end

        % Extract block centered at (i,j)
        r1 = max(1, i-half);
        r2 = min(H, i+half);
        c1 = max(1, j-half);
        c2 = min(W, j+half);

        block = S(r1:r2, c1:c2);

        % Otsu threshold (block-wise)
        T = graythresh(block/255) * 255;  

        % Split block pixels
        Bq = block(block >= T);
        Bo = block(block <  T);

        if isempty(Bq) || isempty(Bo)
            continue;
        end

        % Compute means (μᵇ_q and μᵇ_o)
        uq = mean(Bq);
        uo = mean(Bo);

        % Edge difference
        Db = uq - uo;

        % Classification rule
        if Db > r
            watermark_mask(i,j) = true;
        end
    end
end

edges_clean = imdilate(watermark_mask, strel('disk',5));

%% === 7. Display Results ===
figure;
subplot(3,3,1); imshow(S,[]); title('Structure Image (TV-L1)');
subplot(3,3,2); imshow(S_adj,[]); title('contrast adjustment');
subplot(3,3,3); imshow(S_sharp,[]); title('Enhanced Image');
subplot(3,3,4); imshow(edges); title('Canny Edges');
subplot(3,3,5); imshow(watermark_mask); title('Detected Watermark Edges');
subplot(3,3,6); imshow(edges_clean); title('Dilated Watermark Edges');
subplot(3,3,8); imshow(I); title('AnissoDiffused');

function S = TVL1_denoise(I, lambda, tau, iter)

S = I;
px = zeros(size(I));
py = zeros(size(I));

for k = 1:iter
    % gradient
    [gx, gy] = gradient(S);

    % dual update
    px_new = px + tau * gx;
    py_new = py + tau * gy;

    denom = max(1, sqrt(px_new.^2 + py_new.^2));

    px = px_new ./ denom;
    py = py_new ./ denom;

    % divergence
    div_p = divergence(px, py);

    % primal update
    S = I - lambda * div_p;
end
end

function Iout = anisoDiff(I, num_iter, kappa, lambda)
    I = double(I);
    [rows, cols] = size(I);
    Iout = I;
    for t = 1:num_iter
        % Differences
        north = [diff(Iout,1,1); zeros(1,cols)];
        south = -[zeros(1,cols); diff(Iout,1,1)];
        east  = [diff(Iout,1,2) zeros(rows,1)];
        west  = -[zeros(rows,1) diff(Iout,1,2)];

        % Conductance function (exponential)
        cN = exp(-(north/kappa).^2);
        cS = exp(-(south/kappa).^2);
        cE = exp(-(east/kappa).^2);
        cW = exp(-(west/kappa).^2);

        % Update
        Iout = Iout + lambda*(cN.*north + cS.*south + cE.*east + cW.*west);
    end
end