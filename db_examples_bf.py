
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
    'demo/clean_fg_extracted/9/frames/0000.png',
    'demo/clean_fg_extracted/22/frames/0000.png',
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
        "demo/clean_fg_extracted/14/cropped_video.mp4",
        "demo/clean_bg_extracted/22/cropped_video.mp4",
        "beautiful woman, cinematic lighting",
        "Use Background Video",
        512,
        512,
        "static_fg_sync_bg_visualization_fy/14_22_100fps.mp4",
    ],
    [
        "demo/clean_fg_extracted/14/cropped_video.mp4",
        "demo/clean_bg_extracted/55/cropped_video.mp4",
        "beautiful woman, cinematic lighting",
        "Use Background Video",
        512,
        512,
        "static_fg_sync_bg_visualization_fy/14_55_100fps.mp4",
    ],
    [
        "demo/clean_fg_extracted/15/cropped_video.mp4",
        "demo/clean_bg_extracted/27/cropped_video.mp4",
        "beautiful woman, cinematic lighting",
        "Use Background Video",
        512,
        512,
        
        "static_fg_sync_bg_visualization_fy/15_27_100fps.mp4",
    ],
    [
        "demo/clean_fg_extracted/18/cropped_video.mp4",
        "demo/clean_bg_extracted/23/cropped_video.mp4",
        "beautiful woman, cinematic lighting",
        "Use Background Video",
        512,
        512,
        
        "static_fg_sync_bg_visualization_fy/18_23_100fps.mp4",
    ],
    # [
    #     "demo/clean_fg_extracted/18/cropped_video.mp4",
    #     "demo/clean_bg_extracted/33/cropped_video.mp4",
    #     "beautiful woman, cinematic lighting",
    #     "Use Background Video",
    #     512,
    #     512,
    #     
    #     "static_fg_sync_bg_visualization_fy/18_33_100fps.mp4",
    # ],
    [
        "demo/clean_fg_extracted/22/cropped_video.mp4",
        "demo/clean_bg_extracted/39/cropped_video.mp4",
        "beautiful woman, cinematic lighting",
        "Use Background Video",
        512,
        512,
        
        "static_fg_sync_bg_visualization_fy/22_39_100fps.mp4",
    ],
    # [
    #     "demo/clean_fg_extracted/22/cropped_video.mp4",
    #     "demo/clean_bg_extracted/59/cropped_video.mp4",
    #     "beautiful woman, cinematic lighting",
    #     "Use Background Video",
    #     512,
    #     512,
    #     
    #     "static_fg_sync_bg_visualization_fy/22_59_100fps.mp4",
    # ],
    [
        "demo/clean_fg_extracted/9/cropped_video.mp4",
        "demo/clean_bg_extracted/8/cropped_video.mp4",
        "beautiful woman, cinematic lighting",
        "Use Background Video",
        512,
        512,
        
        "static_fg_sync_bg_visualization_fy/9_8_100fps.mp4",
    ],
    [
        "demo/clean_fg_extracted/9/cropped_video.mp4",
        "demo/clean_bg_extracted/9/cropped_video.mp4",
        "beautiful woman, cinematic lighting",
        "Use Background Video",
        512,
        512,
        
        "static_fg_sync_bg_visualization_fy/9_9_100fps.mp4",
    ],
    [
        "demo/clean_fg_extracted/9/cropped_video.mp4",
        "demo/clean_bg_extracted/10/cropped_video.mp4",
        "beautiful woman, cinematic lighting",
        "Use Background Video",
        512,
        512,
        
        "static_fg_sync_bg_visualization_fy/9_10_100fps.mp4",
    ],
    # [
    #     "demo/clean_fg_extracted/9/cropped_video.mp4",
    #     "demo/clean_bg_extracted/14/cropped_video.mp4",
    #     "beautiful woman, cinematic lighting",
    #     "Use Background Video",
    #     512,
    #     512,
    #     
    #     "static_fg_sync_bg_visualization_fy/9_14_100fps.mp4",
    # ],

]
# background_conditioned_examples = [
#     [
#         "demo/clean_fg_extracted/14/cropped_video.mp4",
#         "demo/clean_bg_extracted/22/cropped_video.mp4",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
#         "static_fg_sync_bg_visualization_fy/14_22_100fps.mp4",
#     ],
#     [
#         "demo/clean_fg_extracted/14/cropped_video.mp4",
#         "demo/clean_bg_extracted/55/cropped_video.mp4",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
#         "static_fg_sync_bg_visualization_fy/14_55_100fps.mp4",
#     ],
#     [
#         "demo/clean_fg_extracted/15/cropped_video.mp4",
#         "demo/clean_bg_extracted/27/cropped_video.mp4",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
        
#         "static_fg_sync_bg_visualization_fy/15_27_100fps.mp4",
#     ],
#     [
#         "demo/clean_fg_extracted/18/cropped_video.mp4",
#         "demo/clean_bg_extracted/23/cropped_video.mp4",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
        
#         "static_fg_sync_bg_visualization_fy/18_23_100fps.mp4",
#     ],
#     # [
#     #     "demo/clean_fg_extracted/18/cropped_video.mp4",
#     #     "demo/clean_bg_extracted/33/cropped_video.mp4",
#     #     "beautiful woman, cinematic lighting",
#     #     "Use Background Video",
#     #     512,
#     #     512,
#     #     
#     #     "static_fg_sync_bg_visualization_fy/18_33_100fps.mp4",
#     # ],
#     [
#         "demo/clean_fg_extracted/22/cropped_video.mp4",
#         "demo/clean_bg_extracted/39/cropped_video.mp4",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
        
#         "static_fg_sync_bg_visualization_fy/22_39_100fps.mp4",
#     ],
#     # [
#     #     "demo/clean_fg_extracted/22/cropped_video.mp4",
#     #     "demo/clean_bg_extracted/59/cropped_video.mp4",
#     #     "beautiful woman, cinematic lighting",
#     #     "Use Background Video",
#     #     512,
#     #     512,
#     #     
#     #     "static_fg_sync_bg_visualization_fy/22_59_100fps.mp4",
#     # ],
#     [
#         "demo/clean_fg_extracted/9/cropped_video.mp4",
#         "demo/clean_bg_extracted/8/cropped_video.mp4",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
        
#         "static_fg_sync_bg_visualization_fy/9_8_100fps.mp4",
#     ],
#     [
#         "demo/clean_fg_extracted/9/cropped_video.mp4",
#         "demo/clean_bg_extracted/9/cropped_video.mp4",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
        
#         "static_fg_sync_bg_visualization_fy/9_9_100fps.mp4",
#     ],
#     [
#         "demo/clean_fg_extracted/9/cropped_video.mp4",
#         "demo/clean_bg_extracted/10/cropped_video.mp4",
#         "beautiful woman, cinematic lighting",
#         "Use Background Video",
#         512,
#         512,
        
#         "static_fg_sync_bg_visualization_fy/9_10_100fps.mp4",
#     ],
#     # [
#     #     "demo/clean_fg_extracted/9/cropped_video.mp4",
#     #     "demo/clean_bg_extracted/14/cropped_video.mp4",
#     #     "beautiful woman, cinematic lighting",
#     #     "Use Background Video",
#     #     512,
#     #     512,
#     #     
#     #     "static_fg_sync_bg_visualization_fy/9_14_100fps.mp4",
#     # ],

# ]
