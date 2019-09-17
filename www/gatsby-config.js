const siteInfo = require("./src/config/site.js");

module.exports = {
  siteMetadata: {
    title: siteInfo.siteTitle,
    author: siteInfo.siteAuthor,
    description: siteInfo.siteDescription,
    siteUrl: siteInfo.siteUrl,
    image: siteInfo.siteImage,
  },
  plugins: [
    `gatsby-plugin-react-helmet`,
    `gatsby-transformer-yaml`,
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        name: `content`,
        path: `${__dirname}/src/content`,
      },
    },
    `gatsby-transformer-sharp`,
    `gatsby-plugin-sharp`,
    {
      resolve: `gatsby-plugin-manifest`,
      options: {
        name: siteInfo.siteTitle,
        short_name: siteInfo.manifestShortName,
        start_url: siteInfo.pathPrefix,
        background_color: siteInfo.manifestBackgroundColor,
        theme_color: siteInfo.manifestThemeColor,
        display: siteInfo.manifestDisplay,
        icon: siteInfo.siteLogo,
      },
    },
    `gatsby-plugin-emotion`,
    {
      resolve: `gatsby-plugin-material-ui`,
        options: {
            stylesProvider: {
              injectFirst: true,
            },
        },
    },
    {
      resolve: `gatsby-plugin-typography`,
      options: {
        pathToConfigModule: `src/utils/typography`
      },
    },
    // this (optional) plugin enables Progressive Web App + Offline functionality
    // To learn more, visit: https://gatsby.dev/offline
    // `gatsby-plugin-offline`,
  ],
}
