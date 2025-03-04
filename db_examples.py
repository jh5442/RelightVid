
bg_samples = [
    'demo/clean_bg_extracted/22/frames/0000.png',
    'demo/clean_bg_extracted/23/frames/0000.png',
    'demo/clean_bg_extracted/27/frames/0000.png',
    'demo/clean_bg_extracted/33/frames/0000.png',
    'demo/clean_bg_extracted/47/frames/0000.png',
    'demo/clean_bg_extracted/39/frames/0000.png',
    'demo/clean_bg_extracted/59/frames/0000.png',
    'demo/clean_bg_extracted/55/frames/0000.png',
    'demo/clean_bg_extracted/58/frames/0000.png',
    'demo/clean_bg_extracted/57/frames/0000.png', #42
    'demo/clean_bg_extracted/8/frames/0000.png',
    'demo/clean_bg_extracted/9/frames/0000.png',
    'demo/clean_bg_extracted/10/frames/0000.png',
    'demo/clean_bg_extracted/14/frames/0000.png',
    'demo/clean_bg_extracted/62/frames/0000.png'
] # 准备大概 15 个 background视频

fg_samples = [
    'demo/clean_fg_extracted/14/frames/0000.png',
    'demo/clean_fg_extracted/15/frames/0000.png',
    'demo/clean_fg_extracted/18/frames/0000.png',
    'demo/clean_fg_extracted/8/frames/0000.png',
    'demo/clean_fg_extracted/1/frames/0000.png',
    # 'demo/clean_fg_extracted/22/frames/0000.png',
    # 'demo/clean_fg_extracted/1/frames/0000.png',
    # 'demo/clean_fg_extracted/2/frames/0000.png',
    # 'demo/clean_fg_extracted/3/frames/0000.png',
    # 'demo/clean_fg_extracted/4/frames/0000.png',
    # 'demo/clean_fg_extracted/5/frames/0000.png',
    # 'demo/clean_fg_extracted/6/frames/0000.png',
    # 'demo/clean_fg_extracted/7/frames/0000.png',
    # 'demo/clean_fg_extracted/8/frames/0000.png',
    # 'demo/clean_fg_extracted/9/frames/0000.png',
    # 'demo/clean_fg_extracted/10/frames/0000.png',
    # 'demo/clean_fg_extracted/11/frames/0000.png',
    # 'demo/clean_fg_extracted/12/frames/0000.png',
    # 'demo/clean_fg_extracted/13/frames/0000.png',
    # 'demo/clean_fg_extracted/16/frames/0000.png',
    # 'demo/clean_fg_extracted/17/frames/0000.png',
    # 'demo/clean_fg_extracted/9/frames/0000.png',
    # 'demo/clean_bg_extracted/39/frames/0000.png',
    # 'demo/clean_bg_extracted/59/frames/0000.png',
    # 'demo/clean_bg_extracted/55/frames/0000.png',
    # 'demo/clean_bg_extracted/58/frames/0000.png',
    # 'demo/clean_bg_extracted/57/frames/0000.png', #42
    # 'demo/clean_bg_extracted/8/frames/0000.png',
    # 'demo/clean_bg_extracted/9/frames/0000.png',
    # 'demo/clean_bg_extracted/10/frames/0000.png',
    # 'demo/clean_bg_extracted/14/frames/0000.png',
    # 'demo/clean_bg_extracted/62/frames/0000.png'
] # 准备大概 15 个 background视频


background_conditioned_examples = [
    [
        1,
        "demo/clean_fg_extracted/14/frames/0000.png",
        "demo/clean_bg_extracted/22/frames/0000.png",
        "beautiful woman, natural lighting",
        "Use Background Video",
        # 512,
        # 512,
        "static_fg_sync_bg_visualization_fy/14_22_100fps.png",
    ],
    [
        2,
        "demo/clean_fg_extracted/14/frames/0000.png",
        "demo/clean_bg_extracted/55/frames/0000.png",
        "beautiful woman, neon dynamic lighting",
        "Use Background Video",
        # 512,
        # 512,
        "static_fg_sync_bg_visualization_fy/14_55_100fps.png",
    ],
    [
        3,
        "demo/clean_fg_extracted/15/frames/0000.png",
        "demo/clean_bg_extracted/27/frames/0000.png",
        "man in suit, tunel lighting",
        "Use Background Video",
        # 512,
        # 512,
        "static_fg_sync_bg_visualization_fy/15_27_100fps.png",
    ],
    [
        4,
        "demo/clean_fg_extracted/18/frames/0000.png",
        "demo/clean_bg_extracted/33/frames/0000.png", # 23->33
        "animated mouse, aesthetic lighting",
        "Use Background Video",
        # 512,
        # 512,
        "static_fg_sync_bg_visualization_fy/18_33_100fps.png",
    ],
    # [
    #     "demo/clean_fg_extracted/18/frames/0000.png",
    #     "demo/clean_bg_extracted/33/frames/0000.png",
    #     "beautiful woman, cinematic lighting",
    #     "Use Background Video",
    #     512,
    #     512,
    #     
    #     "static_fg_sync_bg_visualization_fy/18_33_100fps.png",
    # ],
    [
        5,
        "demo/clean_fg_extracted/22/frames/0000.png",
        "demo/clean_bg_extracted/59/frames/0000.png", # 39 -> 59
        "robot warrior, a sunset background",
        "Use Background Video",
        # 512,
        # 512,
        "static_fg_sync_bg_visualization_fy/22_59_100fps.png",
    ],
    # [
    #     "demo/clean_fg_extracted/22/frames/0000.png",
    #     "demo/clean_bg_extracted/59/frames/0000.png",
    #     "beautiful woman, cinematic lighting",
    #     "Use Background Video",
    #     512,
    #     512,
    #     
    #     "static_fg_sync_bg_visualization_fy/22_59_100fps.png",
    # ],
    
    [
        6,
        "demo/clean_fg_extracted/17/frames/0000.png",
        "demo/clean_bg_extracted/0/frames/0000.png",
        "yellow cat, reflective wet beach",
        "Use Background Video",
        # 512,
        # 512,
        
        "static_fg_sync_bg_visualization_fy/17_0_100fps.png",
    ],
    [
        7,
        "demo/clean_fg_extracted/16/frames/0000.png",
        "demo/clean_bg_extracted/1/frames/0000.png",
        "camera, dock, calm sunset",
        "Use Background Video",
        # 512,
        # 512,
        
        "static_fg_sync_bg_visualization_fy/16_1_100fps.png",
    ],
    [
        8,
        "demo/clean_fg_extracted/9/frames/0000.png",
        "demo/clean_bg_extracted/8/frames/0000.png",
        "astronaut, dim lighting",
        "Use Background Video",
        # 512,
        # 512,
        
        "static_fg_sync_bg_visualization_fy/9_8_100fps.png",
    ],
    [
        9,
        "demo/clean_fg_extracted/9/frames/0000.png",
        "demo/clean_bg_extracted/9/frames/0000.png",
        "astronaut, colorful balloons",
        "Use Background Video",
        # 512,
        # 512,
        "static_fg_sync_bg_visualization_fy/9_9_100fps.png",
    ],
    [
        10,
        "demo/clean_fg_extracted/9/frames/0000.png",
        "demo/clean_bg_extracted/10/frames/0000.png",
        "astronaut, desert landscape",
        "Use Background Video",
        # 512,
        # 512,
        
        "static_fg_sync_bg_visualization_fy/9_10_100fps.png",
    ],
    
    # [
    #     11,
    #     "demo/clean_fg_extracted/7/frames/0000.png",
    #     "demo/clean_bg_extracted/2/frames/0000.png",
    #     "beautiful woman, cinematic lighting",
    #     "Use Background Video",
    #     512,
    #     512,
        
    #     "static_fg_sync_bg_visualization_fy/16_1_100fps.png",
    # ],
    # [
    #     "demo/clean_fg_extracted/9/frames/0000.png",
    #     "demo/clean_bg_extracted/14/frames/0000.png",
    #     "beautiful woman, cinematic lighting",
    #     "Use Background Video",
    #     512,
    #     512,
    #     
    #     "static_fg_sync_bg_visualization_fy/9_14_100fps.png",
    # ],

]
# background_conditioned_examples = [
#     [
#         "demo/clean_fg_extracted/14/frames/0000.png",
#         "demo/clean_bg_extracted/22/frames/0000.png",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
#         "static_fg_sync_bg_visualization_fy/14_22_100fps.png",
#     ],
#     [
#         "demo/clean_fg_extracted/14/frames/0000.png",
#         "demo/clean_bg_extracted/55/frames/0000.png",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
#         "static_fg_sync_bg_visualization_fy/14_55_100fps.png",
#     ],
#     [
#         "demo/clean_fg_extracted/15/frames/0000.png",
#         "demo/clean_bg_extracted/27/frames/0000.png",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
        
#         "static_fg_sync_bg_visualization_fy/15_27_100fps.png",
#     ],
#     [
#         "demo/clean_fg_extracted/18/frames/0000.png",
#         "demo/clean_bg_extracted/23/frames/0000.png",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
        
#         "static_fg_sync_bg_visualization_fy/18_23_100fps.png",
#     ],
#     # [
#     #     "demo/clean_fg_extracted/18/frames/0000.png",
#     #     "demo/clean_bg_extracted/33/frames/0000.png",
#     #     "beautiful woman, cinematic lighting",
#     #     "Use Background Video",
#     #     512,
#     #     512,
#     #     
#     #     "static_fg_sync_bg_visualization_fy/18_33_100fps.png",
#     # ],
#     [
#         "demo/clean_fg_extracted/22/frames/0000.png",
#         "demo/clean_bg_extracted/39/frames/0000.png",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
        
#         "static_fg_sync_bg_visualization_fy/22_39_100fps.png",
#     ],
#     # [
#     #     "demo/clean_fg_extracted/22/frames/0000.png",
#     #     "demo/clean_bg_extracted/59/frames/0000.png",
#     #     "beautiful woman, cinematic lighting",
#     #     "Use Background Video",
#     #     512,
#     #     512,
#     #     
#     #     "static_fg_sync_bg_visualization_fy/22_59_100fps.png",
#     # ],
#     [
#         "demo/clean_fg_extracted/9/frames/0000.png",
#         "demo/clean_bg_extracted/8/frames/0000.png",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
        
#         "static_fg_sync_bg_visualization_fy/9_8_100fps.png",
#     ],
#     [
#         "demo/clean_fg_extracted/9/frames/0000.png",
#         "demo/clean_bg_extracted/9/frames/0000.png",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
        
#         "static_fg_sync_bg_visualization_fy/9_9_100fps.png",
#     ],
#     [
#         "demo/clean_fg_extracted/9/frames/0000.png",
#         "demo/clean_bg_extracted/10/frames/0000.png",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
        
#         "static_fg_sync_bg_visualization_fy/9_10_100fps.png",
#     ],
#     # [
#     #     "demo/clean_fg_extracted/9/frames/0000.png",
#     #     "demo/clean_bg_extracted/14/frames/0000.png",
#     #     "beautiful woman, cinematic lighting",
#     #     "Use Background Video",
#     #     512,
#     #     512,
#     #     
#     #     "static_fg_sync_bg_visualization_fy/9_14_100fps.png",
#     # ],

# ]
