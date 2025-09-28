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
                'install',
            ],
        },
        {
            text: 'Dataflex Selector',
            collapsed: false,
            icon: 'solar:filter-outline',
            prefix: 'selector',
            items: [
                'quickstart',
                'tutorial',
            ],
        },
        {
            text: 'Dataflex Mixer',
            collapsed: false,
            icon: 'solar:shuffle-outline',
            prefix: 'mixer',
            items: [
                'quickstart',
                'tutorial',
            ],
        },
        {
            text: 'Dataflex Weighter',
            collapsed: false,
            icon: 'solar:scale-outline',
            prefix: 'weighter',
            items: [
                'quickstart',
                'tutorial',
            ],
        },
    ],
})
