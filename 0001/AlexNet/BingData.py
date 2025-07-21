from bing_image_downloader import downloader

#cats
downloader.download("cat", limit=50, output_dir='dataset', adult_filter_off=True,
                    force_replace=False,
                    timeout=60)
#dogs
downloader.download("dog", limit=50, output_dir='dataset',adult_filter_off=True,
                    force_replace=False,
                    timeout=60)
