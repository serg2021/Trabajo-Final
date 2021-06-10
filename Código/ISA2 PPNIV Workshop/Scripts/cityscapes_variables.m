cityscapes.cmap_id = zeros(256, 3);
cityscapes.cmap_trainid = zeros(256, 3);

cityscapes.cmap_id(5+1, :) = [111, 74,  0];
cityscapes.cmap_id(6+1, :) = [81,  0, 81];
cityscapes.cmap_id(7+1, :) = [128, 64,128];
cityscapes.cmap_id(8+1, :) = [244, 35,232];
cityscapes.cmap_id(9+1, :) = [250,170,160];
cityscapes.cmap_id(10+1, :) = [230,150,140];
cityscapes.cmap_id(11+1, :) = [70, 70, 70];
cityscapes.cmap_id(12+1, :) = [102,102,156];
cityscapes.cmap_id(13+1, :) = [190,153,153];
cityscapes.cmap_id(14+1, :) = [180,165,180];
cityscapes.cmap_id(15+1, :) = [150,100,100];
cityscapes.cmap_id(16+1, :) = [150,120, 90];
cityscapes.cmap_id(17+1, :) = [153,153,153];
cityscapes.cmap_id(18+1, :) = [153,153,153];
cityscapes.cmap_id(19+1, :) = [250,170, 30];

cityscapes.cmap_id(20+1, :) = [220,220,  0];
cityscapes.cmap_id(21+1, :) = [107,142, 35];
cityscapes.cmap_id(22+1, :) = [152,251,152];
cityscapes.cmap_id(23+1, :) = [70,130,180];
cityscapes.cmap_id(24+1, :) = [220, 20, 60];
cityscapes.cmap_id(25+1, :) = [255,  0,  0];
cityscapes.cmap_id(26+1, :) = [0,  0,142];
cityscapes.cmap_id(27+1, :) = [0,  0, 70];
cityscapes.cmap_id(28+1, :) = [0, 60,100];
cityscapes.cmap_id(29+1, :) = [0,  0, 90];

cityscapes.cmap_id(30+1, :) = [0,  0,110];
cityscapes.cmap_id(31+1, :) = [0, 80,100];
cityscapes.cmap_id(32+1, :) = [0,  0,230];
cityscapes.cmap_id(33+1, :) = [119, 11, 32];

cityscapes.cmap_id = cityscapes.cmap_id / 255;

cityscapes.trainID_to_id = [7 8 11 12 13 17 19:28 31:33] ;

for ii = 1 : length(cityscapes.trainID_to_id)
    cityscapes.cmap_trainid(ii, :) = ...
        cityscapes.cmap_id(cityscapes.trainID_to_id(ii)+1, :);
end

cityscapes.trainID_category = {
    'road'
    'sidewalk'
    'building'
    'wall'
    'fence'
    'pole'
    'traffic light'
    'traffic sign'
    'vegetation'
    'terrain'
    'sky'
    'person'
    'rider'
    'car'
    'truck'
    'bus'
     'train'
     'motorcycle'
     'bicycle'  
     };
%  
% Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
%     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
%     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
%     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
%     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
%     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
%     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
%     Label(  'road'                 ,  7 ,        0 , 'ground'          , 1       , False        , False        , (128, 64,128) ),
%     Label(  'sidewalk'             ,  8 ,        1 , 'ground'          , 1       , False        , False        , (244, 35,232) ),
%     Label(  'parking'              ,  9 ,      255 , 'ground'          , 1       , False        , True         , (250,170,160) ),
%     Label(  'rail track'           , 10 ,      255 , 'ground'          , 1       , False        , True         , (230,150,140) ),
%     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
%     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
%     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
%     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
%     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
%     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
%     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
%     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
%     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
%     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
%     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
%     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
%     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
%     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
%     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
%     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
%     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
%     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
%     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
%     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
%     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
%     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
%     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
%     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),