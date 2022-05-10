classdef HelperFunctions < handle
    %Copyright 2018 The MathWorks, Inc.
    properties
    end
    
    methods(Static)
        
        function pixelLabelColorbar(cmap, classNames)
            % Add a colorbar to the current axis. The colorbar is formatted
            % to display the class names with the color.
            
            colormap(gca,cmap)
            
            % Add colorbar to current figure.
            c = colorbar('peer', gca);
            
            % Use class names for tick marks.
            c.TickLabels = classNames;
            numClasses = size(cmap,1);
            
            % Center tick labels.
            c.Ticks = 1/(numClasses*2):1/numClasses:1;
            
            % Remove tick mark.
            c.TickLength = 0;
        end
        
        function cmap = camvidColorMap()
            % Define the colormap used by CamVid dataset.
            
            cmap = [
                128 128 0     % Environment
                0 255 255     % Building
                192 192 192   % pole
                0 255 0       % road
                255 0 0       % laneDr
                60 40 222     % pavement
                192 128 128   % SignSymbol
                64 0 128      % Car
                64 64 0       % Pedestrian
                0 128 192     % Bicyclist
                ];
            
            % Normalize between [0 1].
            cmap = cmap ./ 255;
        end
        
        function labelIDs = camvidPixelLabelIDs()
            % Return the label IDs corresponding to each class.
            %
            % The CamVid dataset has 32 classes. Group them into 10 classes following
            % the a similar SegNet training methodology [3].
            %
            % The 10 classes are:
            %   "Environment" "Building", "Pole", "Road", "Pavement", "SignSymbol",
            %   "Car", "Pedestrian",  and "Bicyclist".
            %
            % CamVid pixel label IDs are provided as RGB color values. Group them into
            % 11 classes and return as a cell array of M-by-3 matrices. The original
            % CamVid class names are listed alongside each RGB value.
            labelIDs = { ...
                
            
            % "Environment"
            [
            128 128 000; ... % "Tree"
            192 192 000; ... % "VegetationMisc"
            128 128 128; ... % "Sky"
            ]
            
            % "Building"
            [
            000 128 064; ... % "Bridge"
            128 000 000; ... % "Building"
            064 192 000; ... % "Wall"
            064 000 064; ... % "Tunnel"
            192 000 128; ... % "Archway"
            064 064 128; ... % "Fence"
            ]
            
            % "Pole"
            [
            192 192 128; ... % "Column_Pole"
            000 000 064; ... % "TrafficCone"
            ]
            
            % "Road"
            [
            128 064 128; ... % "Road"
            ]
            
            % "Lane"
            [
            128 000 192; ... % "LaneMkgsDriv"
            192 000 064; ... % "LaneMkgsNonDriv"
            ]
            
            % "Pavement"
            [
            000 000 192; ... % "Sidewalk"
            064 192 128; ... % "ParkingBlock"
            128 128 192; ... % "RoadShoulder"
            ]
            
            % "SignSymbol"
            [
            192 128 128; ... % "SignSymbol"
            128 128 064; ... % "Misc_Text"
            000 064 064; ... % "TrafficLight"
            ]
            
            % "Car"
            [
            064 000 128; ... % "Car"
            064 128 192; ... % "SUVPickupTruck"
            192 128 192; ... % "Truck_Bus"
            192 064 128; ... % "Train"
            128 064 064; ... % "OtherMoving"
            ]
            
            % "Pedestrian"
            [
            064 064 000; ... % "Pedestrian"
            192 128 064; ... % "Child"
            064 000 192; ... % "CartLuggagePram"
            064 128 064; ... % "Animal"
            ]
            
            % "Bicyclist"
            [
            000 128 192; ... % "Bicyclist"
            192 000 192; ... % "MotorcycleScooter"
            ]
            
            };
        end
        
        % This is to prepare the data in the beginning
        function prepareData(imageFolder, labelFolder)
            imds = imageDatastore(imageFolder);
            pxds = imageDatastore(labelFolder);
            
            HelperFunctions.resizeCamVidImages(imds, imageFolder);
            HelperFunctions.resizeCamVidLabelImages(pxds, labelFolder);
        end       

        function [imds, pxds] = resizeDataset(imds,imageFolder, pxds, labelFolder)
            % This takes 79 seconds
            imds = HelperFunctions.resizeCamVidImages(imds,imageFolder);
            % This takes 24 seconds
            pxds = HelperFunctions.resizeCamVidPixelLabels(pxds,labelFolder);
        end
            
        function imds = resizeCamVidImages(imds, imageFolder)
            % Resize images to [360 480].
            
            if ~exist(imageFolder,'dir')
                mkdir(imageFolder)
            else
                imds = imageDatastore(imageFolder);
                checkSize = read(imds);
                if size(checkSize) == [360 480 3]
                    disp('Image data already resized')
                    return; % Skip if images already resized
                end
            end
            
            reset(imds)
            while hasdata(imds)
                % Read an image.
                [I,info] = read(imds);
                
                % Resize image.
                I = imresize(I,[360 480]);
                % Write to disk.
                [~, filename, ext] = fileparts(info.Filename);
                imwrite(I, fullfile(imageFolder, [filename  ext]))
            end
            
            imds = imageDatastore(imageFolder);
        end
        
        function pxds = resizeCamVidLabelImages(pxds, labelFolder)
            % Resize pixel label data to [360 480].
            
            if ~exist(labelFolder,'dir')
                mkdir(labelFolder)
            else
                pxds = imageDatastore(labelFolder);
                checkSize = read(pxds);
                if (size(checkSize,1) == 360) && (size(checkSize,2) == 480)
                    disp('Label data already resized')
                    return; % Skip if images already resized
                end
            end
            
            reset(pxds)
            while hasdata(pxds)
                % Read the pixel data.
                [C,info] = read(pxds);
                
                % Resize the data. Use 'nearest' interpolation to
                % preserver label IDs.
                C = imresize(C,[360 480],'nearest');
                
                % Write the data to disk.
                [~, filename, ext] = fileparts(info.Filename);
                imwrite(C,fullfile(labelFolder, [filename ext]))
            end
        end
        
        function pxds = resizeCamVidPixelLabels(pxds, labelFolder)
            % Resize pixel label data to [360 480].
            
            classes = pxds.ClassNames;
            labelIDs = 1:numel(classes);
            if ~exist(labelFolder,'dir')
                mkdir(labelFolder)
            else
                pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
                checkSize = imread(pxds.Files{1});
                if (size(checkSize,1) == 360) && (size(checkSize,2) == 480)
                    disp('Label data already resized')
                    return; % Skip if images already resized
                end
            end
            
            reset(pxds)
            while hasdata(pxds)
                % Read the pixel data.
                [C,info] = read(pxds);
                
                % Convert from categorical to uint8.
                L = uint8(C);
                
                % Resize the data. Use 'nearest' interpolation to
                % preserver label IDs.
                L = imresize(L,[360 480],'nearest');
                
                % Write the data to disk.
                [~, filename, ext] = fileparts(info.Filename);
                imwrite(L,fullfile(labelFolder, [filename ext]))
            end
            
            labelIDs = 1:numel(classes);
            pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
        end

        
        function [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds,labelIDs)
            % Partition CamVid data by randomly selecting 60% of the data for training. The
            % rest is used for testing.
            
            % Set initial random state for example reproducibility.
            rng(0);
            numFiles = numel(imds.Files);
            % Returns a row vector containing a random permutation of the integers from 1 to n inclusive.
            shuffledIndices = randperm(numFiles);
            
            % Use 60% of the images for training.
            N = round(0.60 * numFiles);
            trainingIdx = shuffledIndices(1:N);
            
            % Use the rest for testing.
            testIdx = shuffledIndices(N+1:end);
            
            % Create image datastores for training and test.
            trainingImages = imds.Files(trainingIdx);
            testImages = imds.Files(testIdx);
            imdsTrain = imageDatastore(trainingImages);
            imdsTest = imageDatastore(testImages);
            
            % Extract class and label IDs info
            classes = pxds.ClassNames;
%             labelIDs = 1:numel(pxds.ClassNames);
            
            % Create pixel label datastores for training and test.
            trainingLabels = pxds.Files(trainingIdx);
            testLabels = pxds.Files(testIdx);
            pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
            pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
        end
        
        function showImageMapping(pxds,cmap)
            
            C_rgb = imread(pxds.Files{30});
            C = readimage(pxds, 30);
            
            if size(C_rgb,1) == 720
                B = labeloverlay(C_rgb(600:650,600:650,:),C(600:650,600:650,:),'ColorMap',cmap);
                % Show a ground-truth array
                C(600:650,600:650)
            else
                B = labeloverlay(C_rgb(300:325,300:325,:),C(300:325,300:325,:),'ColorMap',cmap);
                C(300:325,300:325)
            end
            reset(pxds);
            imshow(B,'InitialMagnification',200)
            
        end
        
        function showRotationEffect(imds,pxds,pic_num,degrees)
            
            I = readimage(imds, pic_num);
            Ib = readimage(pxds, pic_num);
            
            cmap = HelperFunctions.camvidColorMap();
            
            IIB = labeloverlay(I, Ib, 'Colormap', cmap, 'Transparency',0.8);
            
            % Convert from categorical to uint8.
            L = uint8(Ib);
            
            Ir = imrotate(I,degrees,'nearest','crop');
            Lr = imrotate(L,degrees,'nearest','crop');
            
            valueset = 1:10;
            
            LB = categorical(Lr, valueset,pxds.ClassNames);
            IB = labeloverlay(Ir, LB, 'Colormap', cmap, 'Transparency',0.8);
            
            figure
            imshowpair(IIB,IB,'montage');
            HelperFunctions.pixelLabelColorbar(cmap, pxds.ClassNames);
            title('Original vs Rotated image')
        end
        
    end
    
    
    
end
