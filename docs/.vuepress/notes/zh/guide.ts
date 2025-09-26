import type { ThemeNote } from 'vuepress-theme-plume'
import { defineNoteConfig } from 'vuepress-theme-plume'

export const Guide: ThemeNote = defineNoteConfig({
    dir: 'guide',
    link: '/guide/',
    sidebar: [
        {
            text: '基本信息',
            collapsed: false,
            icon: 'carbon:idea',
            prefix: 'basicinfo',
            items: [
                'intro',
                'framework',
            ],
        },
        {
            text: '从这里开始',
            collapsed: false,
            icon: 'carbon:idea',
            prefix: 'quickstart',
            items: [
                'install',
                'quickstart',
                'translation',
            ],
        },
    ],
})
