import type { ThemeNote } from 'vuepress-theme-plume'
import { defineNoteConfig } from 'vuepress-theme-plume'

export const Guide: ThemeNote = defineNoteConfig({
    dir: 'guide',
    link: '/guide/',
    sidebar: [
        {
            text: 'Basic Info',
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
            icon: 'carbon:idea',
            prefix: 'selector',
            items: [
                'quickstart',
                'tutorial',
            ],
        },
    ],
})
