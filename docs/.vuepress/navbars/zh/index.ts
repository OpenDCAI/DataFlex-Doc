/**
 * @see https://theme-plume.vuejs.press/config/navigation/ 查看文档了解配置详情
 *
 * Navbar 配置文件，它在 `.vuepress/plume.config.ts` 中被导入。
 */

import { defineNavbarConfig } from 'vuepress-theme-plume'

export const zhNavbar = defineNavbarConfig([
    // { text: '首页', link: '/zh/' },
    // { text: '博客', link: '/zh/blog/' },
    // { text: '标签', link: '/zh/blog/tags/' },
    // { text: '归档', link: '/zh/blog/archives/' },
    {
        text: '指南',
        // link: '/zh/guide/',
        icon: 'icon-park-outline:guide-board',
        items: [
            
            {
                text: '基本信息',
                items: [
                            {
                                text: '简介',
                                link: '/zh/notes/guide/basicinfo/intro.md',
                                icon: 'mdi:tooltip-text-outline',
                                activeMatch: '^/guide/'
                            },
                            {
                                text: '框架设计',
                                link: '/zh/notes/guide/basicinfo/framework.md',
                                icon: 'material-symbols:auto-transmission-sharp',
                                activeMatch: '^/guide/'
                            },
                ]
            },
            {
                text: '快速开始',
                items: [
                    {
                        text: '安装',
                        link: '/zh/notes/guide/quickstart/install.md',
                        icon: 'material-symbols-light:download-rounded',
                        activeMatch: '^/guide/'
                    },
                    {
                        text: '快速开始',
                        link: '/zh/notes/guide/quickstart/quickstart.md',
                        icon: 'solar:flag-2-broken',
                        activeMatch: '^/guide/'
                    }
                ]
            }

        ]
    },
    // {
    //     text: 'API 文档',
    //     link: '/zh/notes/api/1.home.md',
    //     icon: 'material-symbols:article-outline'
    // },

    // {
    //     text: '笔记',
    //     items: [{ text: '示例', link: '/zh/notes/demo/README.md' }]
    // },
])

